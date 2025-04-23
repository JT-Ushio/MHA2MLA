import sys
import inspect
from typing import Optional, Tuple

import torch
from transformers import Cache
from transformers.utils import logging
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    rotate_half,
    LlamaSdpaAttention,
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
        cos_q = cos_q.reshape(1, q.size(2), q.size(1), -1).transpose(1, 2)
        sin_q = sin_q.reshape(1, q.size(2), q.size(1), -1).transpose(1, 2)
        k_idx = k_r_indices[layer_idx].to(k.device)
        cos_k = cos.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        sin_k = sin.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        cos_k = cos_k.reshape(1, k.size(2), k.size(1), -1).transpose(1, 2)
        sin_k = sin_k.reshape(1, k.size(2), k.size(1), -1).transpose(1, 2)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

        return q_embed, k_embed

    return custom_apply_rotary_pos_emb


# Adapted from LlamaAttention.forward
def custom_LlamaSdpaAttention_forward(
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

    query_states = self.q_proj(hidden_states)
    # NOTE: key_states = self.k_proj(hidden_states)
    key_r_states = self.k_r_proj(hidden_states)
    # NOTE: value_states = self.v_proj(hidden_states)
    key_c_states, value_states = self.kv_proj.mha_forward(hidden_states)
    # print('query_states: ', query_states.size())
    # sys.exit()
    key_r_states = key_r_states.view(
        bsz, q_len, self.num_key_value_heads, -1
    ).transpose(1, 2)
    key_c_states = key_c_states.view(
        bsz, q_len, self.num_key_value_heads, -1
    ).transpose(1, 2)
    query_r_states = query_states[..., :self.num_heads*key_r_states.size(-1)]
    query_c_states = query_states[..., self.num_heads*key_r_states.size(-1):]
    query_r_states = query_r_states.view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
    query_c_states = query_c_states.view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
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
    # key_r_states = repeat_kv(key_r_states, self.num_key_value_groups)
    query_r_states, key_r_states = modeling_llama.apply_rotary_pos_emb(
        query_r_states, key_r_states, cos, sin
    )
    query_states = torch.cat([query_r_states, query_c_states], dim=-1)
    # key_c_states = repeat_kv(key_c_states, self.num_key_value_groups)
    key_states = torch.cat([key_r_states, key_c_states], dim=-1)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
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


def mha2mla_llama(q_idx, k_idx):
    LlamaSdpaAttention.forward = custom_LlamaSdpaAttention_forward
    modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb(
        q_idx, k_idx
    )
