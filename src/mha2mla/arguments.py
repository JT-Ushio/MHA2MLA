from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class MHA2MLAModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model"},
    )
    partial_rope_version: str = field(
        default="high",
        metadata={
            "help": "RoPE version to use for partial RoPE in MLA. Options: 'high', 'low', 'uniform', '2-norm'"
        },
    )
    rope_dim_for_mla: int = field(
        default=0, metadata={"help": "Number of rope dimensions per head"}
    )
    uniform_start_point: int = field(
        default=0,
        metadata={
            "help": "Starting point (only used when partial_rope_version='uniform')"
        },
    )
    qk_tensor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-computed QK tensor file, e.g., 'utils/qk_tensor_135M.pth'"
        },
    )
    svd_init_method: str = field(
        default="none",
        metadata={
            "help": "Method for SVD initialization. Options: 'split' or 'joint' or 'none'"
        },
    )
    low_rank: int = field(
        default=8, metadata={"help": "Rank for low-rank approximation in MLA"}
    )
    is_baseline: bool = field(
        default=False,
        metadata={"help": "if the finetuning is the baseline"},
    )
    is_gqa2mha2mla: bool = field(
        default=False,
        metadata={"help": "if the finetuning is GQA2MHA2MLA"},
    )
    is_mla_from_scratch: bool = field(
        default=False,
        metadata={"help": "if the finetuning is from scratch"}
    )

    def __post_init__(self):
        # Call parent class __post_init__ first to ensure any parent validation happens
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # Validate partial_rope_version
        valid_rope_versions = ["high", "low", "uniform", "2-norm"]
        if self.partial_rope_version not in valid_rope_versions:
            raise ValueError(
                f"partial_rope_version must be one of {valid_rope_versions}, got '{self.partial_rope_version}'"
            )

        # Validate svd_init_method
        valid_svd_methods = ["none", "split", "joint", "only_key", "only_value"]
        if self.svd_init_method not in valid_svd_methods:
            raise ValueError(
                f"svd_init_method must be one of {valid_svd_methods}, got '{self.svd_init_method}'"
            )

        # Check bool arguments to avoid conflict
        if self.is_gqa2mha2mla or self.is_mla_from_scratch:
            assert self.is_baseline == False, f"is_baseline must set to False when is_gqa2mha2mla=={self.is_gqa2mha2mla} or is_mla_from_scratch=={self.is_mla_from_scratch}"


@dataclass
class MHA2MLADataArguments:
    is_nanoset: bool = field(
        default=False,
        metadata={
            "help": "Whether to use nanoset dataloader (False means use Huggingface datasets)"
        },
    )
    dataset_folders: Optional[List[str]] = field(
        default=None, metadata={"help": "List of dataset folders to use"}
    )
    dataset_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights for each dataset when mixing multiple datasets"},
    )
    hf_dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Huggingface Dataset name or path"}
    )
    sequence_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length"}
    )


@dataclass
class MHA2MLATrainingArguments(TrainingArguments):
    use_constant_with_warmup_decay_scheduler: bool = field(
        default=False,
        metadata={"help": "Whether to use constant with warmup decay scheduler"},
    )
    is_freeze_non_attn: bool = field(
        default=False,
        metadata={"help": "if the finetuning is freeze non attention parameters"},
    )
