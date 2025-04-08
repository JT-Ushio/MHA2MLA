import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from transformers import HfArgumentParser, DataCollatorForLanguageModeling
from nanotron.data.nanoset import Nanoset
import os
from typing import Dict
import numpy as np
import yaml

from .lr_scheduler import load_scheduler as load_scheduler4constant_with_warmup_decay


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


def load_tokenizer_and_model(
    model_args: ModelArguments, is_mla: bool = False, mla_kwargs: Dict = None
):
    """Load tokenizer and model from configuration."""
    assert (
        model_args.model_name_or_path is not None
    ), "Must provide the path to the model"
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    original_model = mla_kwargs["original_model"]
    if is_mla:
        # prepare instance of mla model
        cfg_RoPE = mla_kwargs.get("RoPE")
        cfg_SVD = mla_kwargs.get("SVD")
        config.RoPE = cfg_RoPE
        config.SVD = cfg_SVD
        model = LlamaForCausalLM(config=config)
        # svd init
        from monkey_patch import state_dict_svd_init

        model.load_state_dict(
            state_dict_svd_init(model, original_model.state_dict()),
        )
    else:
        # directly return the original model
        model = original_model
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
    if model_args.use_constant_with_warmup_decay_scheduler:
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
        "--partial_rope_config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for RoPE.",
    )
    cmd_parser.add_argument(
        "--svd_config",
        type=str,
        required=False,
        default=None,
        help="Path to the YAML configuration file for SVD init.",
    )
    args = cmd_parser.parse_args()
    is_mla = args.svd_config is not None
    config = load_config(args.config_file)
    cfg_RoPE = load_config(args.partial_rope_config)
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, dataset_args = parser.parse_dict(config)
    original_model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path)

    # Monkey Pacth
    if is_mla:
        from monkey_patch import mla_monkey_patch

        mla_monkey_patch(cfg_RoPE)
    else:
        from monkey_patch import partial_rope_monkey_patch

        partial_rope_monkey_patch(cfg_RoPE)

    # Trainer
    if is_mla:
        cfg_SVD = load_config(args.svd_config)
    else:
        cfg_SVD = None
    model, tokenizer = load_tokenizer_and_model(
        model_args,
        is_mla=is_mla,
        mla_kwargs={"RoPE": cfg_RoPE, "SVD": cfg_SVD, "original_model": original_model},
    )
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
