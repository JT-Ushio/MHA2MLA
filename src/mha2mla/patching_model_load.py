import torch
import torch.nn as nn

from patch_func import partial_rope_mask, svd_low_rank_approx


def reorder_matrix_rows(mask, is_cat=False):
    """
    Reorder rows in a matrix based on a binary mask.
    Rows corresponding to 1s in the mask come first, then rows corresponding to 0s.

    Args:
        weight: The weight matrix to reorder
        mask: A binary mask (list or tensor) of length equal to weight.shape[0]

    Returns:
        The reordered weight matrix
    """
    ones_indices = torch.where(mask)[0]
    zeros_indices = torch.where(~mask)[0]
    if is_cat:
        return torch.cat([ones_indices, zeros_indices])
    else:
        return ones_indices, zeros_indices


def patch_model(model, model_args, mha2mla_args):
    """
    Patch a huggingface model by:
    1. Reordering rows in q_proj and k_proj matrices based on a partial-rope mask
    2. Replacing v_proj with a low-rank approximation

    Args:
        model: The Qwen2ForCausalLM model
        mask: A binary mask for reordering q_proj and k_proj
        low_rank: The rank for the low-rank approximation of v_proj
    """
    q_masks, k_masks = partial_rope_mask(model_args, mha2mla_args)

    n_k_head, n_head = model_args.num_key_value_heads, model_args.num_attention_heads
    d_head = model_args.head_dim
    q_idx = []
    k_idx = []
    for layer_idx, layer in enumerate(model.model.layers):
        # 1. Reorder q_proj
        # Get original weights and biases if biases exist
        q_weight = layer.self_attn.q_proj.weight
        q_bias = getattr(layer.self_attn.q_proj, "bias", None)

        # Reorder and update weights and biases if biases exist
        q_mask = q_masks[layer_idx] if len(q_masks.shape) == 2 else q_masks
        q_indices = reorder_matrix_rows(q_mask, is_cat=True)
        layer.self_attn.q_proj.weight.data.copy_(q_weight[q_indices])
        if q_bias is not None:
            layer.self_attn.q_proj.bias.data.copy_(q_bias[q_indices])

        # 2. Reorder k_proj and setup k_r_proj
        # Get original weights and biases if biases exist
        k_weight = layer.self_attn.k_proj.weight
        if mha2mla_args.is_gqa2mha2mla:
            k_weight = (
                k_weight.view(n_k_head, -1, k_weight.size(-1))
                .repeat_interleave(n_head // n_k_head, dim=0)
                .view(-1, k_weight.size(-1))
            )
        k_bias = getattr(layer.self_attn.k_proj, "bias", None)

        # Reorder and update weights and biases if biases exist
        k_mask = k_masks[layer_idx] if len(k_masks.shape) == 2 else k_masks
        k_r_indices, k_c_indices = reorder_matrix_rows(k_mask, is_cat=False)
        k_r_proj = nn.Linear(k_weight.size(1), k_r_indices.size(0), k_bias is not None)
        k_r_proj.weight.data.copy_(k_weight[k_r_indices])
        if k_bias is not None:
            k_r_proj.bias.data.copy_(k_bias[k_r_indices])
        layer.self_attn.k_r_proj = k_r_proj

        # Reorder Q_Norm/K_Norm (e.g., Qwen3) if exist
        if hasattr(layer.self_attn, "q_norm"):
            # TODO: support olmo's Q_Norm/K_Norm
            assert mha2mla_args.partial_rope_version != "2_norm", (
                "Qwen3 does not suppert 2_norm yet."
            )
            norm_indices = reorder_matrix_rows(q_mask[:d_head], is_cat=True)
            qn_weight = layer.self_attn.q_norm.weight
            kn_weight = layer.self_attn.k_norm.weight
            layer.self_attn.q_norm.weight.data.copy_(qn_weight[norm_indices])
            layer.self_attn.k_norm.weight.data.copy_(kn_weight[norm_indices])

        # 3. Setup low-rank kv_proj
        kv_proj = svd_low_rank_approx(
            k_c_weight=k_weight[k_c_indices],
            k_c_bias=k_bias[k_c_indices] if k_bias is not None else None,
            v_weight=layer.self_attn.v_proj.weight,
            v_bias=getattr(layer.self_attn.v_proj, "bias", None),
            d_kv_mid=mha2mla_args.low_rank * model_args.num_key_value_heads,
            method=mha2mla_args.svd_init_method,
        )
        layer.self_attn.kv_proj = kv_proj

        # 4. Delete original k_proj and v_proj
        delattr(layer.self_attn, "k_proj")
        delattr(layer.self_attn, "v_proj")

        d_q_r = n_head * mha2mla_args.rope_dim_for_mla
        q_idx.append(q_indices[:d_q_r])
        k_idx.append(k_r_indices)
        print(f"Layer {layer_idx}: Set up q_proj, k_r_proj, and kv_proj")

        # 5. [Optional] Randomly init if is_mla_from_scratch == True
        if mha2mla_args.is_mla_from_scratch:
            std = model_args.initializer_range
            for name, param in layer.self_attn.named_parameters():
                if "o_proj" in name or "proj" not in name:
                    continue
                if "weight" in name:
                    param.data.normal_(mean=0.0, std=std)
                    print(f"Reinit {name}")
                if "bias" in name:
                    param.data.zero_()
                    print(f"Reinit {name}")

    return model, q_idx, k_idx
