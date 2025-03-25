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
)
from transformers.models.llama import modeling_llama
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_greater_or_equal_2_10



from utils import apply_activation

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


class SvdInit:
    @staticmethod
    def method_I(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down = (U_k[:, :r] + U_v[:, :r]) / 2
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_II(k, v, r=8):
        # Separately decompose W_k_nope and W_v into truncated SVDs, allocating dimensions to each
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down_k = U_k
        W_down_v = U_v
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down_k.t(), W_up_k.t(), W_down_v.t(), W_up_v.t()

    @staticmethod
    def method_III(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        Sigma_k_half = torch.diag(torch.sqrt(S_k))
        Sigma_v_half = torch.diag(torch.sqrt(S_v))
        W_down_k = U_k @ Sigma_k_half
        W_down_v = U_v @ Sigma_v_half
        W_up_k = Sigma_k_half @ V_k.t()
        W_up_v = Sigma_v_half @ V_v.t()
        return W_down_k.t(), W_up_k.t(), W_down_v.t(), W_up_v.t()

    @staticmethod
    def method_IV(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        Sigma_k_half = torch.diag(torch.sqrt(S_k))
        Sigma_v_half = torch.diag(torch.sqrt(S_v))
        W_down_k = U_k @ Sigma_k_half
        W_down_v = U_v @ Sigma_v_half
        W_down = (W_down_k + W_down_v) / 2
        W_up_k = Sigma_k_half @ V_k.t()
        W_up_v = Sigma_v_half @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_V(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        W_down = U_k
        W_down_pseudo_inv = torch.linalg.pinv(W_down)
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.matmul(W_down_pseudo_inv, v)
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_VI(k, v, r=8):
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down = U_v
        W_down_pseudo_inv = torch.linalg.pinv(W_down)
        W_up_k = torch.matmul(W_down_pseudo_inv, k)
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_VII(k, v, r=8):
        # jointly factorize the con-catenated matrix
        U_kv, S_kv, V_kv = torch.svd(torch.cat([k, v], dim=1))
        U_kv, S_kv, V_kv = U_kv[:, :r], S_kv[:r], V_kv[:, :r]
        W_down = U_kv
        split_sizes = [k.size(1), v.size(1)]
        W_up_k, W_up_v = torch.split(V_kv, split_sizes, dim=0)
        W_up_k = torch.diag(S_kv) @ W_up_k.t()
        W_up_v = torch.diag(S_kv) @ W_up_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def init(k, v, svd_method=1, r=8):
        assert k.dtype == v.dtype, "k and v must have the same dtype"
        logger.info(f"Using SVD method {svd_method} with rank {r}")
        original_dtype = k.dtype
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        versions = {
            1: SvdInit.method_I,
            2: SvdInit.method_II,
            3: SvdInit.method_III,
            4: SvdInit.method_IV,
            5: SvdInit.method_V,
            6: SvdInit.method_VI,
            7: SvdInit.method_VII,
        }
        W_down_k, W_up_k, W_down_v, W_up_v = versions[svd_method](k, v, r)
        W_down_k = W_down_k.to(original_dtype)
        W_up_k = W_up_k.to(original_dtype)
        if W_down_v is not None:
            W_down_v = W_down_v.to(original_dtype)
        W_up_v = W_up_v.to(original_dtype)
        return W_down_k, W_up_k, W_down_v, W_up_v


class AutoEncoderV1(nn.Module):
    # Low-rank decomposition of k_nope and v without sharing cache.

    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], nope_mask: torch.Tensor
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.nope_mask = nope_mask
        self.W_down_k = nn.Linear(
            nope_mask.sum().item(),
            config.AE["low_rank"] * config.num_key_value_heads,
            bias=False,
        )
        self.W_down_v = nn.Linear(
            config.num_key_value_heads * config.head_dim,
            config.AE["low_rank"] * config.num_key_value_heads,
            bias=False,
        )
        self.W_up_k = nn.Linear(
            self.W_down_k.out_features,
            nope_mask.sum().item(),
            bias=False,
        )
        self.W_up_v = nn.Linear(
            self.W_down_v.out_features,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )

    def forward(
        self,
        k_r: torch.Tensor,  # [bsz, q_len, rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, nope_dim]
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
        c_k = apply_activation(self.W_down_k(k_nope), self.config.AE["activation_fn"])
        c_v = apply_activation(
            self.W_down_v(value_states), self.config.AE["activation_fn"]
        )
        if past_key_value is not None:
            # change shape for cache(Cache will cat the input on the -2 dim which is the q_len)
            rope_dim = k_r.size(-1)
            k_r = k_r.view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(
                1, 2
            )
            c_kv = torch.cat(
                [
                    c_k.view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(
                        1, 2
                    ),
                    c_v.view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(
                        1, 2
                    ),
                ],
                dim=-1,
            )
            k_r, c_kv = past_key_value.update(k_r, c_kv, self.layer_idx, cache_kwargs)
            k_r = k_r.transpose(1, 2).reshape(bsz, -1, rope_dim)
            c_k, c_v = c_kv.split(
                [
                    self.config.AE["low_rank"],
                    self.config.AE["low_rank"],
                ],
                dim=-1,
            )
            c_k = c_k.transpose(1, 2).reshape(
                bsz, -1, self.config.AE["low_rank"] * self.config.num_key_value_heads
            )
            c_v = c_v.transpose(1, 2).reshape(
                bsz, -1, self.config.AE["low_rank"] * self.config.num_key_value_heads
            )
        k_c = self.W_up_k(c_k)
        value_states = self.W_up_v(c_v)
        key_states = torch.zeros(
            bsz,
            k_c.size(1),
            self.nope_mask.shape[-1],
            device=k_nope.device,
            dtype=k_nope.dtype,
        )
        key_states[..., self.nope_mask] = k_c
        key_states[..., ~self.nope_mask] = k_r
        return key_states, value_states


class AutoEncoderV2(nn.Module):
    # Low-rank decomposition of k_nope and v with shared cache.

    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], nope_mask: torch.Tensor
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.nope_mask = nope_mask
        self.W_down_k = nn.Linear(
            nope_mask.sum().item() + config.num_key_value_heads * config.head_dim,
            config.AE["low_rank"] * config.num_key_value_heads,
            bias=False,
        )
        self.W_up_k = nn.Linear(
            self.W_down_k.out_features,
            nope_mask.sum().item() + config.num_key_value_heads * config.head_dim,
            bias=False,
        )

    def forward(
        self,
        k_r: torch.Tensor,  # [bsz, q_len, rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, nope_dim]
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
        c_kv = apply_activation(self.W_down_k(kv), self.config.AE["activation_fn"])
        if past_key_value is not None:
            # change shape for cache(Cache will cat the input on the -2 dim which is the q_len)
            rope_dim = k_r.size(-1)
            k_r = k_r.view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(
                1, 2
            )
            c_kv = c_kv.view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(
                1, 2
            )
            k_r, c_kv = past_key_value.update(k_r, c_kv, self.layer_idx, cache_kwargs)
            k_r = k_r.transpose(1, 2).reshape(bsz, -1, rope_dim)
            c_kv = c_kv.transpose(1, 2).reshape(
                bsz, -1, self.config.AE["low_rank"] * self.config.num_key_value_heads
            )
        kv = self.W_up_k(c_kv)
        k_c, value_states = kv.split(
            [
                self.nope_mask.sum().item(),
                self.config.num_key_value_heads * self.config.head_dim,
            ],
            dim=-1,
        )
        key_states = torch.zeros(
            bsz,
            k_c.size(1),
            self.nope_mask.shape[-1],
            device=k_nope.device,
            dtype=k_nope.dtype,
        )
        key_states[..., self.nope_mask] = k_c
        key_states[..., ~self.nope_mask] = k_r
        return key_states, value_states


class AutoEncoderV3(nn.Module):
    # Low-rank decomposition of k_nope and v with shared cache. The difference from v2 is that W_down_k and W_up_k are specific to individual heads.

    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], nope_mask: torch.Tensor
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.nope_mask = nope_mask
        self.W_down_k = nn.Linear(
            nope_mask.sum().item() // config.num_key_value_heads + config.head_dim,
            config.AE["low_rank"],
            bias=False,
        )
        self.W_up_k = nn.Linear(
            self.W_down_k.out_features,
            self.W_down_k.in_features,
            bias=False,
        )

    def forward(
        self,
        k_r: torch.Tensor,  # [bsz, q_len, rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, nope_dim]
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
        # kv = torch.cat([k_nope,value_states],dim=-1)
        kv = torch.cat(
            [
                k_nope.view(bsz, q_len, self.config.num_key_value_heads, -1),
                value_states.view(
                    bsz, q_len, self.config.num_key_value_heads, self.config.head_dim
                ),
            ],
            dim=-1,
        )
        c_kv = apply_activation(self.W_down_k(kv), self.config.AE["activation_fn"])
        if past_key_value is not None:
            # change shape for cache(Cache will cat the input on the -2 dim which is the q_len)
            rope_dim = k_r.size(-1)
            k_r = k_r.view(bsz, q_len, self.config.num_key_value_heads, -1).transpose(
                1, 2
            )
            c_kv = c_kv.transpose(1, 2)
            k_r, c_kv = past_key_value.update(k_r, c_kv, self.layer_idx, cache_kwargs)
            k_r = k_r.transpose(1, 2).reshape(bsz, -1, rope_dim)
            c_kv = c_kv.transpose(1, 2)
        kv = self.W_up_k(c_kv)
        k_c, value_states = kv.split(
            [
                self.nope_mask.sum().item()//self.config.num_key_value_heads,
                self.config.head_dim,
            ],
            dim=-1,
        )
        k_c = k_c.reshape(bsz, -1, self.nope_mask.sum().item())
        value_states = value_states.reshape(
            bsz, -1, self.config.num_key_value_heads * self.config.head_dim
        )
        key_states = torch.zeros(
            bsz,
            k_c.size(1),
            self.nope_mask.shape[-1],
            device=k_nope.device,
            dtype=k_nope.dtype,
        )
        key_states[..., self.nope_mask] = k_c
        key_states[..., ~self.nope_mask] = k_r
        return key_states, value_states


AUTO_ENCODER_VERSION_MAP = {
    1: AutoEncoderV1,
    2: AutoEncoderV2,
    3: AutoEncoderV3,
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
        self.nope_mask = IndexForNope.get_index_for_nope(
            config.RoPE,
            head_dim=self.head_dim,
            head_num=self.num_key_value_heads,
            layer_idx=layer_idx,
        )
        self.low_rank = config.AE["low_rank"]

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.nope_mask.sum().item(),
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.W_k_r = nn.Linear(
            self.hidden_size,
            (self.nope_mask == False).sum().item(),
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.auto_encoder = AUTO_ENCODER_VERSION_MAP[config.AE["version"]](
            config, layer_idx, self.nope_mask
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
        query_states = self.q_proj(hidden_states)
        key_states = torch.zeros(
            (bsz, q_len, self.num_key_value_heads * self.head_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        k_r = self.W_k_r(hidden_states)
        key_states[..., ~self.nope_mask] = k_r

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(key_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = self.apply_custom_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        key_states = key_states.transpose(1, 2).reshape(bsz, q_len, -1)
        k_r = key_states[..., ~self.nope_mask]
        k_nope = self.k_proj(hidden_states)
        key_states, value_states = self.auto_encoder(
            k_r=k_r,  # [bsz, q_len, rope_dim]
            k_nope=k_nope,  # [bsz, q_len, nope_dim]
            value_states=self.v_proj(
                hidden_states
            ),  # [bsz, q_len, num_key_value_heads* head_dim]
            cache_kwargs={"sin": sin, "cos": cos, "cache_position": cache_position},
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,  # will become mandatory in v4.46
            **kwargs,
        )  # [bsz, seq_len, num_key_value_heads* head_dim]

        key_states = key_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, -1, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        return query_states, key_states, value_states

    def apply_custom_rotary_pos_emb(self, query_states, key_states, cos, sin):
        if self.config.RoPE["partial_rope_version"] == 4:
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, layer_idx=self.layer_idx
            )
        else:
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        return query_states, key_states

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


def state_dict_svd_init(model, state_dict):
    for layer_idx in range(model.config.num_hidden_layers):
        nope_mask = IndexForNope.get_index_for_nope(
            rope_cfg=model.config.RoPE,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            head_num=model.config.num_key_value_heads,
            layer_idx=layer_idx,
        )
        assert (nope_mask == model.model.layers[layer_idx].self_attn.nope_mask).all()
        W_k = state_dict.pop(f"model.layers.{layer_idx}.self_attn.k_proj.weight").t()
        W_v = state_dict.pop(f"model.layers.{layer_idx}.self_attn.v_proj.weight").t()
        state_dict[f"model.layers.{layer_idx}.self_attn.W_k_r.weight"] = W_k[
            ..., ~nope_mask
        ].t()
        W_down_k, W_up_k, W_down_v, W_up_v = SvdInit.init(
            W_k[..., nope_mask],
            W_v,
            svd_method=model.config.SVD["method"],
            r=model.config.SVD["low_rank"] * model.config.num_key_value_heads,
        )
        state_dict[f"model.layers.{layer_idx}.self_attn.W_down_k.weight"] = W_down_k
        state_dict[f"model.layers.{layer_idx}.self_attn.W_up_k.weight"] = W_up_k
        if not model.model.layers[layer_idx].self_attn.is_share_W_down:
            state_dict[f"model.layers.{layer_idx}.self_attn.W_down_v.weight"] = W_down_v
        state_dict[f"model.layers.{layer_idx}.self_attn.W_up_v.weight"] = W_up_v
    return state_dict


@classmethod
def custom_load_pretrained_model(
    cls,
    model,
    state_dict,
    loaded_keys,
    resolved_archive_file,
    pretrained_model_name_or_path,
    ignore_mismatched_sizes=False,
    sharded_metadata=None,
    _fast_init=True,
    low_cpu_mem_usage=False,
    device_map=None,
    offload_folder=None,
    offload_state_dict=None,
    dtype=None,
    hf_quantizer=None,
    keep_in_fp32_modules=None,
    gguf_path=None,
    weights_only=True,
):
    if all(["W_k_r" not in k for k in state_dict.keys()]) and isinstance(
        model.model, modeling_llama.LlamaPreTrainedModel
    ):
        # replace the original llama weights with the mla weights
        state_dict = state_dict_svd_init(model, state_dict)
        loaded_keys = list(state_dict.keys())
    old_k_r_weight = model.model.layers[0].self_attn.W_k_r.weight
    outputs = cls.original_load_pretrained_model(
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes,
        sharded_metadata,
        _fast_init,
        low_cpu_mem_usage,
        device_map,
        offload_folder,
        offload_state_dict,
        dtype,
        hf_quantizer,
        keep_in_fp32_modules,
        gguf_path,
        weights_only,
    )
    new_k_r_weight = model.model.layers[0].self_attn.W_k_r.weight
    assert not (old_k_r_weight == new_k_r_weight).all()
    return outputs


def ae_patch_func_hf(rope_cfg=None):
    modeling_llama.LLAMA_ATTENTION_CLASSES = {
        "eager": CustomLlamaAttention,
        "flash_attention_2": CustomLlamaFlashAttention2,
        "sdpa": CustomLlamaSdpaAttention,
    }

    # if not hasattr(modeling_llama.LlamaPreTrainedModel, 'original_load_pretrained_model'):
    #     modeling_llama.LlamaPreTrainedModel.original_load_pretrained_model = modeling_llama.LlamaPreTrainedModel._load_pretrained_model
    #     modeling_llama.LlamaPreTrainedModel._load_pretrained_model = custom_load_pretrained_model

    if rope_cfg is not None:
        # replace apply_rotary_pos_emb function in llama model
        from ..partial_rope.patch_func_hf import create_custom_apply_rotary_pos_emb_hf

        modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_hf(
            rope_cfg
        )
