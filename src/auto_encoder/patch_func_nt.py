from typing import Any, Dict, Optional, Tuple
import nanotron.config
import nanotron.config
import nanotron.config.config
import nanotron.config.models_config
import nanotron.trainer
import torch
from torch import nn
import torch.distributed as dist
from packaging.version import Version
from pathlib import Path
from dataclasses import dataclass, field
import nanotron
from nanotron.s3_checkpoints import check_path_is_local
from safetensors.torch import safe_open
from tqdm import tqdm
from nanotron.config import (
    ParallelismArgs,
    LlamaConfig,
    ExistingCheckpointInit,
    RandomInit,
    SpectralMupInit,
)
from nanotron.generation.generate_store import AttachableStore
from torch.nn.parallel import DistributedDataParallel
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.logging import log_rank
from nanotron import logging
from nanotron.distributed import get_global_rank
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.models import NanotronModel
from nanotron.models import llama
from nanotron.models.llama import (
    CoreAttention,
    logger,
)
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.parallel.parameters import NanotronParameter
from nanotron.serialize import weights as nt_weights
from nanotron.serialize.weights import (
    get_checkpoint_version,
    read_checkpoint_version_from_shard_file,
    CheckpointVersionFromShardFileException,
    load_sharded_param_latest,
)
from nanotron.parallel import ParallelContext
from nanotron.serialize.utils import (
    ObjectType,
    get_exp_tp_pp_rank_and_size_from,
    get_path,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ..mla.NopeIndex import IndexForNope
from ..mla.svd_low_rank import SvdInit
from .utils import apply_activation


@dataclass
class CustomLlamaConfig(LlamaConfig):
    RoPE: Dict = field(default_factory=dict)
    AE: Dict = field(default_factory=dict)


@dataclass
class CustomModelArgs(nanotron.config.config.ModelArgs):
    model_config: CustomLlamaConfig


@dataclass
class CustomConfig(nanotron.config.config.Config):
    """Main configuration class"""

    model: CustomModelArgs

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
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: Optional[int],
        nope_mask: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.tp_mode = (
            self.parallel_config.tp_mode
            if self.parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        self.tp_linear_async_communication = (
            self.parallel_config.tp_linear_async_communication
            if self.parallel_config is not None
            else False
        )
        self.layer_idx = layer_idx
        self.nope_mask = nope_mask
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.W_down_k = TensorParallelColumnLinear(
            nope_mask.sum().item(),
            config.AE["low_rank"] * config.num_key_value_heads,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_down_v = TensorParallelColumnLinear(
            config.num_key_value_heads * self.d_v,
            config.AE["low_rank"] * config.num_key_value_heads,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_up_k = TensorParallelColumnLinear(
            self.W_down_k.out_features,
            nope_mask.sum().item(),
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_up_v = TensorParallelColumnLinear(
            self.W_down_v.out_features,
            config.num_key_value_heads * self.d_v,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def forward(
        self,
        k_r: torch.Tensor,  # [bsz, q_len, rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, nope_dim]
        value_states: torch.Tensor,  # [bsz, q_len, n_local_kv_heads* d_v]
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]
    ]:  # [bsz, q_len, n_local_kv_heads* d_v]
        bsz, q_len, _ = k_nope.size()
        c_k = apply_activation(self.W_down_k(k_nope), self.config.AE["activation_fn"])
        c_v = apply_activation(
            self.W_down_v(value_states), self.config.AE["activation_fn"]
        )
        k_c = self.W_up_k(c_k)
        value_states = self.W_up_v(c_v)
        key_states = torch.zeros(
            bsz,
            q_len,
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
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: Optional[int],
        nope_mask: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.tp_mode = (
            self.parallel_config.tp_mode
            if self.parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        self.tp_linear_async_communication = (
            self.parallel_config.tp_linear_async_communication
            if self.parallel_config is not None
            else False
        )
        self.layer_idx = layer_idx
        self.nope_mask = nope_mask
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.W_down_k = TensorParallelColumnLinear(
            nope_mask.sum().item() + config.num_key_value_heads * self.d_v,
            config.AE["low_rank"] * config.num_key_value_heads,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_up_k = TensorParallelColumnLinear(
            self.W_down_k.out_features,
            nope_mask.sum().item() + config.num_key_value_heads * self.d_v,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def forward(
        self,
        k_r: torch.Tensor,  # [bsz, q_len, rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, nope_dim]
        value_states: torch.Tensor,  # [bsz, q_len, n_local_kv_heads* d_v]
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]
    ]:  # [bsz, q_len, n_local_kv_heads* d_v]
        bsz, q_len, _ = k_nope.size()
        kv = torch.cat([k_nope, value_states], dim=-1)
        c_kv = apply_activation(self.W_down_k(kv), self.config.AE["activation_fn"])
        kv = self.W_up_k(c_kv)
        k_c, value_states = kv.split(
            [self.nope_mask.sum().item(), self.config.num_key_value_heads * self.d_v],
            dim=-1,
        )
        key_states = torch.zeros(
            bsz,
            q_len,
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
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: Optional[int],
        nope_mask: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.tp_mode = (
            self.parallel_config.tp_mode
            if self.parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        self.tp_linear_async_communication = (
            self.parallel_config.tp_linear_async_communication
            if self.parallel_config is not None
            else False
        )
        self.layer_idx = layer_idx
        self.nope_mask = nope_mask
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.W_down_k = TensorParallelColumnLinear(
            nope_mask.sum().item() // config.num_key_value_heads + self.d_v,
            config.AE["low_rank"],
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_up_k = TensorParallelColumnLinear(
            self.W_down_k.out_features,
            self.W_down_k.in_features,
            bias=False,
            pg=self.tp_pg,
            mode=self.tp_mode,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def forward(
        self,
        k_r: torch.Tensor,  # [bsz, q_len, rope_dim]
        k_nope: torch.Tensor,  # [bsz, q_len, nope_dim]
        value_states: torch.Tensor,  # [bsz, q_len, n_local_kv_heads* d_v]
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]
    ]:  # [bsz, q_len, n_local_kv_heads* d_v]
        bsz, q_len, _ = k_nope.size()
        kv = torch.cat(
            [
                k_nope.view(bsz, q_len, self.config.num_key_value_heads, -1),
                value_states.view(
                    bsz, q_len, self.config.num_key_value_heads, self.d_v
                ),
            ],
            dim=-1,
        )
        c_kv = apply_activation(self.W_down_k(kv), self.config.AE["activation_fn"])
        kv = self.W_up_k(c_kv)
        k_c, value_states = kv.split(
            [self.nope_mask.sum().item() // self.config.num_key_value_heads, self.d_v],
            dim=-1,
        )
        k_c = k_c.reshape(bsz, -1, self.nope_mask.sum().item())
        value_states = value_states.reshape(
            bsz, -1, self.config.num_key_value_heads * self.d_v
        )
        key_states = torch.zeros(
            bsz,
            q_len,
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


class CustomCausalSelfAttention(nn.Module, AttachableStore):
    def __init__(
        self,
        config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        assert (
            not config.rope_interleaved
        ), "MLA Causal attention does not support interleaved RoPE"
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = (
            config.num_attention_heads != config.num_key_value_heads
        )  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup
        self.layer_idx = layer_idx
        self.low_rank = config.AE["low_rank"]
        self.tp_mode = (
            self.parallel_config.tp_mode
            if self.parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        self.tp_linear_async_communication = (
            self.parallel_config.tp_linear_async_communication
            if self.parallel_config is not None
            else False
        )

        self.nope_mask = IndexForNope.get_index_for_nope(
            config.RoPE,
            head_dim=self.d_qk,
            head_num=self.n_local_kv_heads,
            layer_idx=layer_idx,
        )
        # TODO @thomasw21: refactor so that we store that default in a single place.

        # build the slice config for self.qkv for save/load
        # shard are done within the contiguous chunk

        # TODO(kunhao): We want to have only one version per device and not one version per layer.
        self.rotary_embedding = llama.LlamaRotaryEmbedding(
            dim=self.d_qk,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.rope_interleaved = config.rope_interleaved
        self.auto_encoder = AUTO_ENCODER_VERSION_MAP[config.AE["version"]](
            config=self.config,
            parallel_config=self.parallel_config,
            tp_pg=self.tp_pg,
            layer_idx=self.layer_idx,
            nope_mask=self.nope_mask,
        )
        self.init_w_q()
        self.init_w_k()
        self.init_w_v()
        self.init_w_o()

        # NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)
        self.flash_rotary_embedding = FlashRotaryEmbedding(
            dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = (
            config.max_position_embeddings
        )  # TODO @nouamane: compute based on free memory, because in rope we can surpass max_position_embeddings

    def init_w_q(self):
        self.q_proj = TensorParallelColumnLinear(
            self.d_model,
            self.config.num_attention_heads * self.d_qk,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def init_w_k(self):
        self.k_proj = TensorParallelColumnLinear(
            self.d_model,
            self.nope_mask.sum().item(),
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )
        self.W_k_r = TensorParallelColumnLinear(
            self.d_model,
            (self.nope_mask == False).sum().item(),
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def init_w_v(self):
        self.v_proj = TensorParallelColumnLinear(
            self.d_model,
            self.n_local_kv_heads * self.d_v,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
            tp_recompute_allgather=self.parallel_config.tp_recompute_allgather,
        )

    def init_w_o(self):
        self.o_proj = TensorParallelRowLinear(
            self.config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=self.tp_pg,
            mode=self.tp_mode,
            bias=False,
            async_communication=self.tp_linear_async_communication,
        )

    def get_query_states(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        query_states = self.q_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk]
        q_length, batch_size, _ = query_states.shape
        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )  # [batch_size, seq_length, n_local_q_heads, d_qk]
        return query_states

    def get_key_value_states(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        k_r = self.W_k_r(hidden_states).transpose(
            0, 1
        )  # [seq_length, batch_size, rope_dim]
        k_nope = self.k_proj(hidden_states).transpose(
            0, 1
        )  # [seq_length, batch_size, nope_dim]
        value_states = self.v_proj(hidden_states).transpose(
            0, 1
        )  # [seq_length, batch_size, n_local_kv_heads * d_v]
        key_states, value_states = self.auto_encoder(k_r, k_nope, value_states)
        key_states = key_states.view(
            key_states.size(0),
            key_states.size(1),
            self.n_local_kv_heads,
            self.d_qk,
        )
        value_states = value_states.view(
            value_states.size(0),
            value_states.size(1),
            self.n_local_kv_heads,
            self.d_v,
        )
        return key_states, value_states

    def forward(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        from flash_attn import bert_padding
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        batch_size, q_length = sequence_mask.shape

        query_states = self.get_query_states(hidden_states, sequence_mask)
        key_states, value_states = self.get_key_value_states(
            hidden_states, sequence_mask
        )

        store = self.get_local_store()  # In fact, collections.defaultdict?
        if store is not None:  # Inference case
            assert False, "Not implemented"

        else:  # Training case
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
            cos, sin = self.rotary_embedding(value_states, position_ids)

            if self.config.RoPE["partial_rope_version"] == 4:
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, layer_idx=self.layer_idx
                )
            else:
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            q_sequence_mask = sequence_mask
            kv_sequence_mask = sequence_mask

            kv_length = key_states.shape[1]
            # [batch_size, seq_length, num_heads, d_qk]
            # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
            query_states = query_states.view(
                batch_size * q_length, self.n_local_q_heads, self.d_qk
            )  # [batch_size * q_length, self.n_heads, d_qk]

            key_states = key_states.view(
                batch_size * kv_length, self.n_local_kv_heads, self.d_qk
            )  # [batch_size * kv_length, self.n_heads, d_qk]
            value_states = value_states.view(
                batch_size * kv_length, self.n_local_kv_heads, self.d_v
            )  # [batch_size * kv_length, self.n_heads, d_v]

            attention_output = self.attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                q_sequence_mask=q_sequence_mask,
                kv_sequence_mask=kv_sequence_mask,
            )

        attention_output = (
            attention_output.contiguous()
            .view(batch_size, q_length, self.n_local_q_heads * self.d_v)
            .transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {
            "hidden_states": output,
            "sequence_mask": sequence_mask,
        }


def custom_load_weights(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """Load weights from a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {
        id(module): f"{module_name}." for module_name, module in model.named_modules()
    }
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    checkpoint_version: Optional[Version] = None

    filtered_state_dict = (
        filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    )
    param_shard_metadata = {}
    missing_keys = []
    for name, param_or_buffer in tqdm(
        filtered_state_dict.items(),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Loading weights",
    ):
        # NOTE: extract how does the current model parameter are sharded
        # so that we can load optimizer checkpoints in this way
        param_shard_metadata[name] = {}
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter):
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[
                        sharded_info.global_ranks
                    ]
                    group_rank = dist.get_rank(group)

                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank),
                    parallel_context=parallel_context,
                )
                # TODO @nouamane: do we consider exp_size=1 expert_sharded?
                is_expert_sharded = sharded_info.is_expert_sharded(parallel_context)
            else:
                exp_tp_pp_rank_and_size = None
                is_expert_sharded = False

            path = get_path(
                base_name,
                type=ObjectType.MODEL,
                exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                prefix=param_root_folder,
                is_expert_sharded=is_expert_sharded,
            )

            if path.exists():
                with safe_open(path, framework="pt", device=str(param.device)) as fi:
                    # TODO @thomasw21: Choose only a slice if we switch the TP topology
                    param_or_buffer[:] = fi.get_tensor("data")

            elif not path.parent.exists():
                missing_keys.append(name)
                # raise ValueError(
                #     f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                #     f"Couldn't find folder {path.parent} in checkpoint at {root_folder}"
                # )
            else:
                # Let's assume that the topology changed and the param is sharded.
                # We search for all the files from the shards, concatenate the "unsharded" tensor
                # and load the specific shard we're interested in.
                if not param.is_sharded:
                    raise ValueError(
                        f"`{name}` is not a sharded parameter. It's possible you were expecting {path} to exist."
                    )
                # TODO @thomasw21: Make so that we don't need to code this logic somewhere else than in `get_path`
                sharded_info = param.get_sharded_info()
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(
                    path.parent.glob(f"{ObjectType.MODEL.value}_{suffix}*.safetensors")
                )
                if len(shards_path) <= 0:
                    raise ValueError(
                        f"Could not find any shards {ObjectType.MODEL.value}_{suffix}*.safetensors in {path.parent}."
                        f"If you notice `.safetensors` in the middle of the name of some of the checkpoints files. You need to run `scripts/fix_checkpoint_bad_naming.py`."
                    )

                if checkpoint_version is None:
                    checkpoint_version = get_checkpoint_version(
                        parallel_context, root_folder, param_save_path=shards_path[0]
                    )
                else:
                    current_checkpoint_version = None
                    try:
                        current_checkpoint_version = (
                            read_checkpoint_version_from_shard_file(
                                param_save_path=shards_path[0]
                            )
                        )
                    except CheckpointVersionFromShardFileException:
                        # The checkpoint version is read from the meta file
                        current_checkpoint_version = checkpoint_version
                    finally:
                        assert (
                            current_checkpoint_version == checkpoint_version
                        ), f"Checkpoint version mismatch at {shards_path[0]}."

                if checkpoint_version <= CHECKPOINT_VERSION:
                    load_sharded_param_latest(
                        param_or_buffer=param_or_buffer,
                        sharded_info=sharded_info,
                        shards_path=shards_path,
                        param_shard_metadata=param_shard_metadata[name],
                    )
                else:
                    raise ValueError(
                        f"Unsupported checkpoint version {checkpoint_version}"
                    )

        else:
            raise NotImplementedError(
                f"Parameters {param} should be a NanotronParameter"
            )

    low_rank = model.config.AE["low_rank"] * model.config.num_key_value_heads
    logger.info(f"Low rank: {low_rank}")
    legal_keys = [
        "W_k_r.weight",
        "k_proj.weight",
        "W_down_k.weight",
        "W_up_k.weight",
        "W_down_v.weight",
        "W_up_v.weight",
        "v_proj.weight",
        "q_proj.weight",
    ]
    for missing_key in missing_keys:
        layer_idx = int(missing_key.split(".")[2])
        attn_module_prefix = f"model.decoder.{layer_idx}.pp_block.attn"
        attn_module = model.model.decoder[layer_idx].pp_block.attn
        base_name = f"model.decoder.{layer_idx}.pp_block.attn.qkv_proj.weight"
        path = get_path(
            base_name,
            type=ObjectType.MODEL,
            exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
            prefix=param_root_folder,
            is_expert_sharded=True,
        )
        with safe_open(path, framework="pt", device=str(param.device)) as fi:
            qkv_proj = fi.get_tensor("data").t()
        q_proj, k_proj, v_proj = qkv_proj.split(
            [
                attn_module.n_local_q_heads * attn_module.d_qk,
                attn_module.n_local_kv_heads * attn_module.d_qk,
                attn_module.n_local_kv_heads * attn_module.d_v,
            ],
            dim=-1,
        )
        if missing_key.endswith("W_down_k.weight"):
            dtype = filtered_state_dict[
                f"{attn_module_prefix}.auto_encoder.W_down_k.weight"
            ].dtype
            in_features = attn_module.auto_encoder.W_down_k.in_features
            out_features = attn_module.auto_encoder.W_down_k.out_features
            filtered_state_dict[f"{attn_module_prefix}.auto_encoder.W_down_k.weight"][
                :
            ] = torch.nn.init.xavier_uniform_(
                torch.empty(out_features, in_features, dtype=dtype)
            )
            filtered_state_dict[f"{attn_module_prefix}.auto_encoder.W_up_k.weight"][
                :
            ] = torch.nn.init.xavier_uniform_(
                torch.empty(in_features, out_features, dtype=dtype)
            )
        elif missing_key.endswith("W_down_v.weight"):
            dtype = filtered_state_dict[
                f"{attn_module_prefix}.auto_encoder.W_down_v.weight"
            ].dtype
            in_features = attn_module.auto_encoder.W_down_v.in_features
            out_features = attn_module.auto_encoder.W_down_v.out_features
            filtered_state_dict[f"{attn_module_prefix}.auto_encoder.W_down_v.weight"][
                :
            ] = torch.nn.init.xavier_uniform_(
                torch.empty(out_features, in_features, dtype=dtype)
            )
            filtered_state_dict[f"{attn_module_prefix}.auto_encoder.W_up_v.weight"][
                :
            ] = torch.nn.init.xavier_uniform_(
                torch.empty(in_features, out_features, dtype=dtype)
            )
        elif missing_key.endswith("q_proj.weight"):
            filtered_state_dict[f"{attn_module_prefix}.q_proj.weight"][:] = q_proj.t()
        elif missing_key.endswith("v_proj.weight"):
            filtered_state_dict[f"{attn_module_prefix}.v_proj.weight"][:] = v_proj.t()
        elif missing_key.endswith("W_k_r.weight"):
            filtered_state_dict[f"{attn_module_prefix}.W_k_r.weight"][:] = (
                k_proj[:, ~attn_module.nope_mask]
            ).t()
        elif missing_key.endswith("k_proj.weight"):
            filtered_state_dict[f"{attn_module_prefix}.k_proj.weight"][:] = (
                k_proj[:, attn_module.nope_mask]
            ).t()
        elif any([missing_key.endswith(suffix) for suffix in legal_keys]):
            continue
        else:
            raise ValueError(
                f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                f"Couldn't find folder {path.parent} in checkpoint at {root_folder}."
                f"Missing key: {missing_key}"
            )

    return param_shard_metadata


def ae_patch_func_nt(rope_cfg=None):
    llama.CausalSelfAttention = CustomCausalSelfAttention
    if not hasattr(nt_weights, "original_load_weights"):
        nt_weights.original_load_weights = nt_weights.load_weights
        nt_weights.load_weights = custom_load_weights
    nanotron.config.models_config.LlamaConfig = CustomLlamaConfig
    nanotron.trainer.CONFIG_TO_MODEL_CLASS.update(
        {"CustomLlamaConfig": nanotron.trainer.CONFIG_TO_MODEL_CLASS["LlamaConfig"]}
    )

    nanotron.serialize.load_weights = custom_load_weights
    nanotron.trainer.load_weights = custom_load_weights
    from ..partial_rope.patch_func import create_custom_apply_rotary_pos_emb

    llama.LlamaRotaryEmbedding.apply_rotary_pos_emb = (
        create_custom_apply_rotary_pos_emb(rope_cfg)
    )
    from ..mla.NopeIndex import IndexForNope

    if rope_cfg["partial_rope_version"] == 4:
        IndexForNope._qk_tensor_cache = torch.load(rope_cfg["qk_tensor_path"])
        IndexForNope._qk_tensor_path = rope_cfg["qk_tensor_path"]
