from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class MHA2MLAModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model"},
    )


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
