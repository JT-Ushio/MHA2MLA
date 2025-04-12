from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
import datasets
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import HfArgumentParser, DataCollatorForLanguageModeling
from nanotron.data.nanoset import Nanoset
import os
from typing import Dict, List
import numpy as np
import yaml

from lr_scheduler import load_scheduler as load_scheduler4constant_with_warmup_decay


@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_constant_with_warmup_decay_scheduler: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: str = None
    tokenizer_name_or_path: str = None
    save_initial_model: bool = False


@dataclass
class DataArguments:
    is_nanoset: bool = False
    dataset_folders: List[str] = None
    dataset_weights: List[float] = None
    dataset_name_or_path: str = None
    sequence_length: int = 2048


class CustomNanoset(Nanoset):
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: The input ids wrapped in a dictionary
        """
        item = super().__getitem__(idx)
        return item


def load_dataset(dataset_args, training_args, tokenizer):
    """Load dataset from configuration."""
    tokenizer.model_max_length = dataset_args.sequence_length
    if dataset_args.is_nanoset:
        dataset_folders = dataset_args.dataset_folders
        dataset_weights = dataset_args.dataset_weights
        sequence_length = dataset_args.sequence_length
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        global_batch_size = (
            training_args.per_device_train_batch_size
            * world_size
            * training_args.gradient_accumulation_steps
        )
        dataset = CustomNanoset(
            dataset_folders=dataset_folders,
            sequence_length=sequence_length,
            dataset_weights=dataset_weights,
            token_size=token_size,
            train_split_num_samples=global_batch_size * training_args.max_steps,
        )
    else:
        import datasets

        dataset = datasets.load_dataset(
            dataset_args.dataset_name_or_path, split="train"
        )

    return dataset


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_tokenizer_and_model(model_args: ModelArguments):
    """Load tokenizer and model from configuration."""
    assert model_args.model_name_or_path is not None, (
        "Must provide the path to the model"
    )
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_optimizer_scheduler(model, training_args, model_args):
    """Load optimizer and scheduler from configuration."""
    optimizer_name = training_args.optim
    if "adam" in optimizer_name:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=training_args.learning_rate,
            betas=(
                training_args.adam_beta1,
                training_args.adam_beta2,
            ),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            fused=bool(training_args.optim == "adamw_torch_fused"),
        )
    else:
        raise ValueError(f"Unknown optimizer factory {optimizer_name}")
    if training_args.use_constant_with_warmup_decay_scheduler:
        lr_scheduler = load_scheduler4constant_with_warmup_decay(
            optimizer, training_args
        )
    else:
        from transformers import get_scheduler

        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
    return optimizer, lr_scheduler


def main():
    import argparse

    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    cmd_parser.add_argument(
        "--is_auto_encoder",
        action="store_true",
        help="Whether to use auto_encoder.",
    )
    args = cmd_parser.parse_args()
    config = load_config(args.config_file)
    parser = HfArgumentParser((CustomTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, dataset_args = parser.parse_dict(config)

    # Monkey Pacth
    if args.is_auto_encoder:
        from monkey_patch import ae_patch_func_hf

        ae_patch_func_hf()

    # Trainer
    model, tokenizer = load_tokenizer_and_model(model_args)
    if training_args.bf16:
        model = model.to(dtype=torch.bfloat16)
    elif training_args.fp16:
        model = model.to(dtype=torch.float16)

    train_dataset = load_dataset(dataset_args, training_args, tokenizer)
    resume_from_checkpoint = training_args.resume_from_checkpoint
    optimizer, lr_scheduler = load_optimizer_scheduler(model, training_args, model_args)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator,
    )
    # train
    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint)
    else:
        if int(os.getenv("LOCAL_RANK", 0)) == 0 and model_args.save_initial_model:
            trainer._save_checkpoint()
        trainer.train()


if __name__ == "__main__":
    main()
