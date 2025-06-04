import inspect
from typing import Callable, Optional, Tuple

import torch
from transformers import Cache
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    rotate_half,
    eager_attention_forward,
    Qwen3MoeAttention,
)

logger = logging.get_logger(__name__)


def create_custom_apply_rotary_pos_emb(q_r_indices, k_r_indices):
    # Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        # TODO: access layer_idx only for 2-norm
        # Get the calling frame
        frame = inspect.currentframe().f_back
        # Get the 'self' argument of the caller
        attention_module = frame.f_locals["self"]
        # Access the layer_idx
        layer_idx = attention_module.layer_idx

        # NOTE: cos = cos.unsqueeze(unsqueeze_dim)
        # NOTE: sin = sin.unsqueeze(unsqueeze_dim)
        q_idx = q_r_indices[layer_idx].to(q.device)
        cos_q = cos.repeat(1, 1, q.size(1)).index_select(-1, q_idx)
        sin_q = sin.repeat(1, 1, q.size(1)).index_select(-1, q_idx)
        cos_q = cos_q.reshape(cos_q.size(0), q.size(2), q.size(1), -1).transpose(1, 2)
        sin_q = sin_q.reshape(sin_q.size(0), q.size(2), q.size(1), -1).transpose(1, 2)

        k_idx = k_r_indices[layer_idx].to(k.device)
        cos_k = cos.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        sin_k = sin.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        cos_k = cos_k.reshape(cos_k.size(0), k.size(2), k.size(1), -1).transpose(1, 2)
        sin_k = sin_k.reshape(sin_k.size(0), k.size(2), k.size(1), -1).transpose(1, 2)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

        return q_embed, k_embed

    return custom_apply_rotary_pos_emb


def custom_Qwen3MoeAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # NOTE: value_states = self.v_proj(hidden_states)
    key_c_states, value_states = self.kv_proj.mha_forward(hidden_states)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    kv_shape = (*input_shape, value_states.size(1), -1)
    key_r_states = self.k_r_proj(hidden_states).view(kv_shape)#.transpose(1, 2)
    key_c_states = key_c_states.view(kv_shape)#.transpose(1, 2)
    key_states = self.k_norm(
        torch.cat([key_r_states, key_c_states], dim=-1)
    ).transpose(1, 2)
    key_r_states = key_states[..., : key_r_states.size(-1)]
    key_c_states = key_states[..., key_r_states.size(-1) :]
    query_r_states = query_states[..., : key_r_states.size(-1)]
    query_c_states = query_states[..., key_r_states.size(-1) :]

    cos, sin = position_embeddings
    query_r_states, key_r_states = modeling_qwen3_moe.apply_rotary_pos_emb(
        query_r_states, key_r_states, cos, sin
    )

    query_states = torch.cat([query_r_states, query_c_states], dim=-1)
    key_states = torch.cat([key_r_states, key_c_states], dim=-1)

    # NOTE: the code below has not been modified.
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def mha2mla_qwen3_moe(q_idx, k_idx):
    Qwen3MoeAttention.forward = custom_Qwen3MoeAttention_forward
    modeling_qwen3_moe.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb(
        q_idx, k_idx
    )
