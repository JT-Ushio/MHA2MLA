from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import copy
import sys


import math
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    logger,
    LlamaRotaryEmbedding,
    repeat_kv,
    rotate_half,
)
from transformers.models.llama import modeling_llama
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_greater_or_equal_2_10


def apply_activation(x: torch.Tensor, activation_fn: str):
    if activation_fn is not None:
        activation_fn = activation_fn.lower()
    if activation_fn is None:
        return x
    elif activation_fn == "relu":
        return F.relu(x)
    elif activation_fn == "sigmoid":
        return torch.sigmoid(x)
    elif activation_fn == "tanh":
        return torch.tanh(x)
    elif activation_fn == "leaky_relu":
        return F.leaky_relu(x)
    elif activation_fn == "softmax":
        return F.softmax(x, dim=-1)
    elif activation_fn == "silu" or activation_fn == "swish":
        return F.silu(x)
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")


class IndexForNope:
    _qk_tensor_path = None
    _qk_tensor_cache = None

    @staticmethod
    def get_index_for_nope_v0(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        nope_mask = torch.zeros((head_dim), dtype=torch.bool)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v1(rope_cfg, **kwargs):
        keep_dim = rope_cfg["top_k_rope_dim"]
        head_dim = kwargs["head_dim"]
        if keep_dim <= 0:
            nope_mask = torch.ones((head_dim), dtype=torch.bool)
        elif keep_dim >= head_dim:
            nope_mask = torch.zeros((head_dim), dtype=torch.bool)
        else:
            half = head_dim // 2
            nope_mask = torch.ones((half), dtype=torch.bool)
            nope_mask[:keep_dim] = False
            nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v2(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        indices_to_remove = torch.arange(
            rope_cfg["uniform_start_point"], head_dim, rope_cfg["uniform_step"]
        )
        nope_mask = torch.ones(head_dim, dtype=torch.bool)
        nope_mask[indices_to_remove] = False
        return nope_mask

    @staticmethod
    def get_index_for_nope_v3(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        top_k_dim, last_k_dim = rope_cfg["top_k_rope_dim"], rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        assert top_k_dim + last_k_dim <= half
        nope_mask = torch.zeros((half), dtype=torch.bool)
        nope_mask[top_k_dim : half - last_k_dim] = True
        nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v4(rope_cfg, **kwargs):
        if (
            IndexForNope._qk_tensor_cache is None
            or rope_cfg["qk_tensor_path"] != IndexForNope._qk_tensor_path
        ):
            with open(rope_cfg["qk_tensor_path"], "rb") as fin:
                IndexForNope._qk_tensor_cache = torch.load(
                    fin
                )  # [layer_num, k_head_num, head_dim//2]
                IndexForNope._qk_tensor_path = rope_cfg["qk_tensor_path"]
                assert len(IndexForNope._qk_tensor_cache.shape) == 3
        qk_tensor = IndexForNope._qk_tensor_cache
        layer_idx = kwargs["layer_idx"]
        top_k_dim = rope_cfg["top_k_rope_dim"]
        topk_indices = torch.topk(qk_tensor[layer_idx], k=top_k_dim, dim=1)[1]
        nope_mask = torch.ones_like(qk_tensor[layer_idx], dtype=torch.bool)
        nope_mask.scatter_(1, topk_indices, False)
        nope_mask = torch.cat([nope_mask, nope_mask], dim=-1)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v5(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        last_k_rope_dim = rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        nope_mask = torch.ones((half), dtype=torch.bool)
        nope_mask[half - last_k_rope_dim : half] = False
        nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope(rope_cfg, **kwargs):
        logger.info(f"rope_cfg: {rope_cfg}")
        version = rope_cfg["partial_rope_version"]
        versions = {
            0: IndexForNope.get_index_for_nope_v0,
            1: IndexForNope.get_index_for_nope_v1,
            2: IndexForNope.get_index_for_nope_v2,
            3: IndexForNope.get_index_for_nope_v3,
            4: IndexForNope.get_index_for_nope_v4,
            5: IndexForNope.get_index_for_nope_v5,
        }
        index_func = versions[version]
        nope_mask = index_func(rope_cfg, **kwargs)
        nope_mask = nope_mask.to(dtype=torch.bool)
        if version == 4:
            nope_mask = nope_mask.reshape(-1)
        else:
            nope_mask = nope_mask.repeat(repeats=(kwargs["head_num"],))
        return nope_mask


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AutoEncoder(nn.Module):
    # Low-rank decomposition of k_nope and v with shared cache.

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int]):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.nope_mask_for_k = IndexForNope.get_index_for_nope(
            config.RoPE,
            head_dim=self.head_dim,
            head_num=self.num_key_value_heads,
            layer_idx=layer_idx,
        )
        self.nope_mask_for_q = (
            self.nope_mask_for_k.view(config.num_key_value_heads, self.head_dim)
            .repeat_interleave(
                config.num_attention_heads // config.num_key_value_heads, dim=0
            )
            .reshape(-1)
        )
        self.head_nope_dim = (
            self.nope_mask_for_k.sum().item() // config.num_key_value_heads
        )
        self.head_rope_dim = self.head_dim - self.head_nope_dim
        self.W_down = nn.Linear(
            config.num_key_value_heads * self.head_nope_dim + config.num_key_value_heads * self.head_dim,
            config.auto_encoder["low_rank_per_head"] * config.num_key_value_heads,
            bias=False,
        )
        self.W_up = nn.Linear(
            self.W_down.out_features,
            self.W_down.in_features,
            bias=False,
        )
        if "enable_norm" in config.auto_encoder and config.auto_encoder["enable_norm"]:
            self.kv_layernorm = RMSNorm(
                config.auto_encoder["low_rank_per_head"] * config.num_key_value_heads
            )

    def forward(
        self,
        k_rope: torch.Tensor,  # [bsz, num_key_value_heads, q_len, head_rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, num_key_value_heads* head_nope_dim]
        value_states: torch.Tensor,  # [bsz, q_len, num_key_value_heads* head_dim]
        cache_kwargs: dict = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]
    ]:  # [bsz, q_len, num_key_value_heads* head_dim]
        bsz, q_len, _ = k_nope.size()
        kv = torch.cat([k_nope, value_states], dim=-1)
        c_kv = apply_activation(
            self.W_down(kv), self.config.auto_encoder["activation_fn"]
        )
        c_kv = c_kv.view(bsz, q_len, 1, -1).transpose(
            1, 2
        )  # [bsz, 1, q_len, low_rank_per_head* self.num_key_value_heads]
        if hasattr(self, "kv_layernorm"):
            c_kv = self.kv_layernorm(c_kv)
        if past_key_value is not None:
            k_rope, c_kv = past_key_value.update(
                k_rope, c_kv, self.layer_idx, cache_kwargs
            )
        kv = self.W_up(c_kv)
        k_nope, value_states = kv.split(
            [
                self.nope_mask_for_k.sum().item(),
                self.config.num_key_value_heads * self.config.head_dim,
            ],
            dim=-1,
        )  # [bsz, seq_len, key_value_heads* head_dim]
        return k_rope, k_nope, value_states


AUTO_ENCODER_VERSION_MAP = {
    2: AutoEncoder,
}


class CustomLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.low_rank_per_head = config.auto_encoder["low_rank_per_head"]

        self.auto_encoder = AUTO_ENCODER_VERSION_MAP[config.auto_encoder["version"]](
            config, layer_idx
        )
        self.nope_mask_for_k = self.auto_encoder.nope_mask_for_k
        self.nope_mask_for_q = self.auto_encoder.nope_mask_for_q
        self.head_rope_dim = self.auto_encoder.head_rope_dim
        self.head_nope_dim = self.auto_encoder.head_nope_dim
        self.W_q_rope = nn.Linear(
            self.hidden_size,
            self.num_heads * self.auto_encoder.head_rope_dim,
            bias=config.attention_bias,
        )
        self.W_q_nope = nn.Linear(
            self.hidden_size,
            self.num_heads * self.auto_encoder.head_nope_dim,
            bias=config.attention_bias,
        )
        self.W_k_rope = nn.Linear(
            self.hidden_size,
            (~self.nope_mask_for_k).sum().item(),
            bias=config.attention_bias,
        )
        self.W_k_nope = nn.Linear(
            self.hidden_size,
            self.nope_mask_for_k.sum().item(),
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def get_qkv_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        assert not self.config.pretraining_tp > 1, "not support pretraining_tp>1"
        # prepare query_states and k_r
        q_rope = self.W_q_rope(hidden_states)
        q_nope = self.W_q_nope(hidden_states)
        k_rope = self.W_k_rope(hidden_states)
        k_nope = self.W_k_nope(hidden_states)
        value_states = self.v_proj(hidden_states)

        q_rope = q_rope.view(bsz, q_len, self.num_heads, self.head_rope_dim).transpose(
            1, 2
        )
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.head_nope_dim).transpose(
            1, 2
        )
        k_rope = k_rope.view(
            bsz, q_len, self.num_key_value_heads, self.head_rope_dim
        ).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        q_rope, k_rope = self.apply_custom_rotary_pos_emb(
            q_rope, k_rope, cos, sin
        )  # [bsz, num_heads, q_len, head_rope_dim]
        k_rope, k_nope, value_states = self.auto_encoder(
            k_rope=k_rope,
            k_nope=k_nope,
            value_states=value_states,
            cache_kwargs=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        k_nope = k_nope.view(
            bsz, q_len, self.num_key_value_heads, self.head_nope_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query_states = torch.cat(
            [q_rope, q_nope], dim=-1
        )  # [bsz, num_heads, q_len, head_dim]
        key_states = torch.cat(
            [k_rope, k_nope], dim=-1
        )  # [bsz, num_key_value_heads, q_len, head_dim]
        return query_states, key_states, value_states

    def apply_custom_rotary_pos_emb(self, q_rope, k_rope, cos, sin, unsqueeze_dim=1):
        assert unsqueeze_dim == 1, "Unsqueeze_dim should be 1"
        rotary_mask_for_k = (~self.nope_mask_for_k).view(
            1, self.num_key_value_heads, 1, self.head_dim
        )
        cos = cos.unsqueeze(unsqueeze_dim).repeat_interleave(
            self.num_key_value_heads, unsqueeze_dim
        )
        sin = sin.unsqueeze(unsqueeze_dim).repeat_interleave(
            self.num_key_value_heads, unsqueeze_dim
        )
        bsz, num_heads, q_len, _ = cos.size()
        rotary_mask_for_k = rotary_mask_for_k.expand_as(cos)
        cos = cos[rotary_mask_for_k].view(bsz, num_heads, q_len, -1)
        sin = sin[rotary_mask_for_k].view(bsz, num_heads, q_len, -1)
        k_embed = (k_rope * cos) + (rotate_half(k_rope) * sin)
        cos = cos.repeat_interleave(
            self.num_key_value_groups, unsqueeze_dim
        )  # [1, num_heads, q_len, head_rope_dim]
        sin = sin.repeat_interleave(
            self.num_key_value_groups, unsqueeze_dim
        )  # [1, num_heads, q_len, head_rope_dim]
        q_embed = (q_rope * cos) + (rotate_half(q_rope) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.get_qkv_states(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CustomLlamaFlashAttention2(CustomLlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.get_qkv_states(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
        )

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CustomLlamaSdpaAttention(CustomLlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.get_qkv_states(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def ae_patch_func_hf(rope_cfg=None):
    modeling_llama.LLAMA_ATTENTION_CLASSES = {
        "eager": CustomLlamaAttention,
        "flash_attention_2": CustomLlamaFlashAttention2,
        "sdpa": CustomLlamaSdpaAttention,
    }



