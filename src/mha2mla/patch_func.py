import torch
import torch.nn as nn


def partial_rope_mask(model_args, mha2mla_args):
    """
    Generate different types of masks for partial rotary position embeddings (RoPE)
    based on configuration settings.
    Returns:
        Appropriate mask tensor based on the specified version
    """
    n_head = model_args.num_attention_heads
    n_k_head = model_args.num_key_value_heads
    d_head = model_args.hidden_size // n_head
    d_head_half = d_head // 2
    rope_dim_for_mla = mha2mla_args.rope_dim_for_mla
    rope_dim_for_mla_half = rope_dim_for_mla // 2
    rope_version = mha2mla_args.partial_rope_version
    mask = torch.zeros(d_head)

    def select_high_frequency(mask):
        """
        Select high-frequency components (first rope_dim_for_mla dimensions)
        Returns:
            mask: Binary mask with 1s for the first rope_dim_for_mla dimensions
        """
        mask[:rope_dim_for_mla_half] = 1
        mask[d_head_half : d_head_half + rope_dim_for_mla_half] = 1
        q_masks = mask.repeat(n_head)
        k_masks = mask.repeat(n_k_head)
        return q_masks, k_masks

    def select_low_frequency(mask):
        """
        Select low-frequency components (last rope_dim_for_mla dimensions)
        Returns:
            mask: Binary mask with 1s for the last rope_dim_for_mla dimensions
        """
        mask[d_head - rope_dim_for_mla_half :] = 1
        mask[d_head_half - rope_dim_for_mla_half : d_head_half] = 1
        q_masks = mask.repeat(n_head)
        k_masks = mask.repeat(n_k_head)
        return q_masks, k_masks

    def select_uniform_frequency(mask, start_point):
        """
        Select uniformly distributed dimensions for RoPE
        Returns:
            mask: Binary mask with 1s at uniformly spaced positions
        """
        step = d_head // rope_dim_for_mla
        assert d_head_half % step == 0, "rope_dim_for_mla must be greater than 0"

        for i in range(start_point, d_head, step):
            mask[i] = 1
        q_masks = mask.repeat(n_head)
        k_masks = mask.repeat(n_k_head)
        return q_masks, k_masks

    def select_2norm_frequency(mask, rope_dim_for_mla):
        """
        Select dimensions based on 2-norm frequency importance
        Returns:
            mask: Binary mask with 1s for the top rope_dim_for_mla dimensions by 2-norm
        """
        # This is a placeholder implementation since the exact 2-norm selection
        # method was not detailed in the comments. In practice, this would
        # require statistics from the weight matrices to determine importance.
        with open(mha2mla_args.qk_tensor_path, "rb") as fin:
            qk_norm_rank = torch.load(fin, weights_only=True)

        k_masks = qk_norm_rank < rope_dim_for_mla_half
        q_masks = k_masks.repeat_interleave(n_head // n_k_head, dim=1)
        k_masks = k_masks.view(k_masks.size(0), -1)
        q_masks = q_masks.view(q_masks.size(0), -1)
        return q_masks, k_masks

    if rope_version == "high":
        return select_high_frequency(mask)
    elif rope_version == "low":
        return select_low_frequency(mask)
    elif rope_version == "uniform":
        return select_uniform_frequency(mask, mha2mla_args.uniform_start_point)
    elif rope_version == "2-norm":
        return select_2norm_frequency(d_head, rope_dim_for_mla)


class LowRankKVLinear(nn.Module):
    """
    A low-rank approximation of a linear layer.
    Instead of storing a full matrix W of shape (out_features, in_features),
    it stores two matrices: down of shape (in_features, low_rank) and
    up of shape (low_rank, out_features), such that W ≈ up.T @ down.T
    """

    def __init__(self, d_kv_in, d_kv_mid=0, d_k_mid=0, d_k_out=0, d_v_mid=0, d_v_out=0, bias=None):
        super().__init__()
        # TODO: add activations after down_kv
        self.down_kv = nn.Linear(in_features=d_kv_in, out_features=d_kv_mid, bias=False)
        if d_k_mid and d_k_out:
            self.up_k = nn.Linear(in_features=d_k_mid, out_features=d_k_out, bias=bias)
        if d_v_mid and d_v_out:
            self.up_v = nn.Linear(in_features=d_v_mid, out_features=d_v_out, bias=bias)
        self.d_kv_mid = d_kv_mid
        self.d_k_mid, self.d_v_mid = d_k_mid, d_v_mid
        self.d_k_out, self.d_v_out = d_k_out, d_v_out

    def reset_parameters(
        self,
        down_kv_weight,
        up_k_weight=None,
        up_v_weight=None,
        up_k_bias=None,
        up_v_bias=None,
    ):
        self.down_kv.weight.data.copy_(down_kv_weight)
        if up_k_weight is not None:
            self.up_k.weight.data.copy_(up_k_weight)
        if up_v_weight is not None:
            self.up_v.weight.data.copy_(up_v_weight)
        if up_k_bias is not None:
            self.up_k.bias.data.copy_(up_k_bias)
        if up_v_bias is not None:
            self.up_v.bias.data.copy_(up_v_bias)

    def mha_forward(self, x):
        # x: (batch_size, seq_len, in_features)
        kv = self.down_kv(x)
        d_kv, d_k, d_v = self.d_kv_mid, self.d_k_mid, self.d_v_mid
        if hasattr(self, "up_k"):
            k = self.up_k(kv) if d_kv == d_k else self.up_k(kv[:d_k])
        else:
            k = kv[..., : self.d_k_out]
        if hasattr(self, "up_v"):
            v = self.up_v(kv) if d_kv == d_v else self.up_v(kv[-d_v:])
        else:
            v = kv[..., -self.d_v_out :]
        return k, v

    def mla_forward(self, x, q_hid, o_hid):
        kv = self.down_kv(x)
        # TODO: Triton kernel
        return kv


class LowRankKVLinear2(nn.Module):
    """
    A low-rank approximation of a linear layer.
    Instead of storing a full matrix W of shape (out_features, in_features),
    it stores two matrices: down of shape (in_features, low_rank) and
    up of shape (low_rank, out_features), such that W ≈ up.T @ down.T
    """

    def __init__(
        self,
        d_mid=0,
        d_k_in=0,
        d_k_out=0,
        k_approx=False,
        d_v_in=0,
        d_v_out=0,
        v_approx=False,
        kv_joint=False,
        bias=None,
    ):
        super().__init__()
        # TODO: add activations after down_kv
        if kv_joint:
            self.down_kv = nn.Linear(in_features=d_k_in + d_v_in, out_features=d_mid, bias=False)
        if not kv_joint and k_approx:
            self.down_k = nn.Linear(in_features=d_k_in, out_features=d_mid, bias=False)
        if not kv_joint and v_approx:
            self.down_v = nn.Linear(in_features=d_v_in, out_features=d_mid, bias=False)

        if k_approx:
            self.up_k = nn.Linear(in_features=d_mid, out_features=d_k_out, bias=bias)
        else:
            self.up_k = nn.Linear(in_features=d_k_in, out_features=d_k_out, bias=bias)
        if v_approx:
            self.up_v = nn.Linear(in_features=d_mid, out_features=d_v_out, bias=bias)
        else:
            self.up_v = nn.Linear(in_features=d_v_in, out_features=d_v_out, bias=bias)

    def reset_parameters(
        self,
        down_kv_weight=None,
        up_k_weight=None,
        up_v_weight=None,
        up_k_bias=None,
        up_v_bias=None,
    ):
        if down_kv_weight is not None:
            self.down_kv.weight.data.copy_(down_kv_weight)
        if up_k_weight is not None:
            self.up_k.weight.data.copy_(up_k_weight)
        if up_v_weight is not None:
            self.up_v.weight.data.copy_(up_v_weight)
        if up_k_bias is not None:
            self.up_k.bias.data.copy_(up_k_bias)
        if up_v_bias is not None:
            self.up_v.bias.data.copy_(up_v_bias)

    def mha_forward(self, x):
        # x: (batch_size, seq_len, in_features)
        kv = self.down_kv(x)
        d_kv, d_k, d_v = self.d_kv_mid, self.d_k_mid, self.d_v_mid
        if hasattr(self, "up_k"):
            k = self.up_k(kv) if d_kv == d_k else self.up_k(kv[:d_k])
        else:
            k = kv[..., : self.d_k_out]
        if hasattr(self, "up_v"):
            v = self.up_v(kv) if d_kv == d_v else self.up_v(kv[-d_v:])
        else:
            v = kv[..., -self.d_v_out :]
        return k, v

    def mla_forward(self, x, q_hid, o_hid):
        kv = self.down_kv(x)
        # TODO: Triton kernel
        return kv


def SVD(X, r):
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    U, S, V = U[:, :r], S[:r], V[:r, :]
    U @= torch.diag(S)
    return V, U


def svd_low_rank_approx(k_c_weight, k_c_bias, v_weight, v_bias, d_kv_mid, method):
    d_k_c, d_v, d_kv_in = k_c_weight.size(0), v_weight.size(0), v_weight.size(1)
    if method == "only_key":
        down_k, up_k = SVD(k_c_weight, d_kv_mid)
        up_v = None
        down_kv = torch.cat([down_k, v_weight], dim=0)
        d_k_mid = d_kv_mid
        d_kv_mid += d_v
        d_v_mid = d_v = 0
    elif method == "only_value":
        down_v, up_v = SVD(v_weight, d_kv_mid)
        up_k = None
        down_kv = torch.cat([k_c_weight, down_v])
        d_v_mid = d_kv_mid
        d_kv_mid += d_k_c
        d_k_mid = d_k_c = 0
    elif method == "split":
        down_k, up_k = SVD(k_c_weight, d_kv_mid)
        down_v, up_v = SVD(v_weight, d_kv_mid)
        down_kv = torch.cat([down_k, down_v])
        d_k_mid = d_v_mid = d_kv_mid
        d_kv_mid *= 2
    elif method == "joint":
        joint_kv = torch.cat([k_c_weight, v_weight])
        down_kv, up_kv = SVD(joint_kv, d_kv_mid)
        up_k, up_v = up_kv.split([d_k_c, d_v])
        d_k_mid = d_v_mid = d_kv_mid
    elif method == "none":
        down_kv = torch.cat([k_c_weight, v_weight])
        up_k = up_v = None
        d_kv_mid = d_k_c + d_v
        d_k_mid = d_v_mid = 0

    has_bias = k_c_bias is not None
    kv_proj = LowRankKVLinear(d_kv_in, d_kv_mid, d_k_mid, d_k_c, d_v_mid, d_v, has_bias)
    kv_proj.reset_parameters(down_kv, up_k, up_v, k_c_bias, v_bias)
    return kv_proj
