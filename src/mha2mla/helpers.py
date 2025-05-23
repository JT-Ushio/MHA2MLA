import os
import sys
import math
import logging
from types import SimpleNamespace
from functools import partial

import torch
import numpy as np
from nanotron.data.nanoset import Nanoset
from nanotron.logging import log_rank
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler


logger = logging.getLogger(__name__)


def freeze_non_attn_weights(model):
    print('......freeze_non_attn_weights......')
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "attn" in name or "norm" in name: 
            param.requires_grad = True


def load_optimizer_scheduler(model, train_args):
    """Load optimizer and scheduler from configuration."""
    optimizer_name = train_args.optim
    # TODO: remove weight_decay for bias and norm
    if "adam" in optimizer_name:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=train_args.learning_rate,
            betas=(
                train_args.adam_beta1,
                train_args.adam_beta2,
            ),
            eps=train_args.adam_epsilon,
            weight_decay=train_args.weight_decay,
            fused=bool(train_args.optim == "adamw_torch_fused"),
        )
    else:
        raise ValueError(f"Unknown optimizer factory {optimizer_name}")

    if train_args.use_constant_with_warmup_decay_scheduler:
        lr_scheduler = lr_scheduler_builder(
            optimizer=optimizer,
            lr_scheduler_args=SimpleNamespace(**train_args.lr_scheduler_kwargs),
            total_training_steps=train_args.max_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=train_args.warmup_steps,
            num_training_steps=train_args.max_steps,
        )
    return optimizer, lr_scheduler


def load_dataset(dataset_args, train_args, tokenizer):
    """Load dataset from configuration."""
    tokenizer.model_max_length = dataset_args.sequence_length
    if dataset_args.is_nanoset:
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_batch_size = (
            train_args.per_device_train_batch_size
            * world_size
            * train_args.gradient_accumulation_steps
        )
        dataset = Nanoset(
            dataset_folders=dataset_args.dataset_folders,
            sequence_length=dataset_args.sequence_length,
            dataset_weights=dataset_args.dataset_weights,
            token_size=token_size,
            train_split_num_samples=global_batch_size * train_args.max_steps,
        )
    else:
        import datasets

        dataset = datasets.load_dataset(
            dataset_args.dataset_name_or_path, split="train"
        )

    return dataset


# Copyright [2025] [nanotron]
# Adapted from nanotron.helpers.lr_scheduler_builder
# Changes made:
# - Changed optimizer.get_base_optimizer().param_groups to optimizer.param_groups
def lr_scheduler_builder(
    optimizer: Optimizer, lr_scheduler_args, total_training_steps: int
):
    if lr_scheduler_args.lr_decay_steps is None:
        lr_decay_steps = total_training_steps
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_steps -= lr_scheduler_args.lr_warmup_steps
        if lr_scheduler_args.lr_decay_starting_step is not None:
            lr_decay_steps -= lr_scheduler_args.lr_decay_starting_step
    else:
        lr_decay_steps = lr_scheduler_args.lr_decay_steps

    if lr_scheduler_args.lr_decay_starting_step is None:
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_starting_step = lr_scheduler_args.lr_warmup_steps
        else:
            lr_decay_starting_step = 0
    else:
        lr_decay_starting_step = lr_scheduler_args.lr_decay_starting_step

    def lr_lambda(current_step: int, initial_lr: float):
        """
        current_step: current training step
        initial_lr: the learning rate of a parameter group

        More info on initial_lr:
        And in standard parameterization, lr_lambda only takes a single learning rate.
        But in µTransfer, each parameter has a custom learning rate (custom_lr = lr_scheduler_args.learning_rate * scaling_factor),
        so each parameter group has a custom lr_lambda function.

        LR Scheduling function, it has from 2 up to 4 phases:
        - warmup,
        - optional: constant (if lr_decay_starting_step is set)
        - decay
        - optional: constant (if lr_decay_steps and/or lr_decay_starting_step are set)
        Warmup starts at lr=0 and ends at `lr=lr`
        Then it stays constant at lr if lr_decay_starting_step is set and larger than lr_warmup_steps
        Then it decays until `min_decay_lr` for lr_decay_steps if set, else: (total_training_steps - lr_warmup_steps or lr_decay_starting_step)
        Then it stays constant at min_decay_lr if lr_decay_starting_step is set and total_training_steps is larger)
        """
        # No warmup or decay
        if lr_scheduler_args.lr_warmup_steps == 0 and lr_decay_steps == 0:
            return initial_lr

        # Warmup phase
        elif (
            lr_scheduler_args.lr_warmup_style is not None
            and current_step <= lr_scheduler_args.lr_warmup_steps
        ):
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = (
                    initial_lr
                    * current_step
                    / max(lr_scheduler_args.lr_warmup_steps, 1)
                )
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = lr_scheduler_args.learning_rate
            else:
                raise ValueError(
                    f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}"
                )

        # Optional constant phase at learning_rate
        elif current_step < lr_decay_starting_step:
            lmbda = initial_lr

        # Decay phase
        elif (
            lr_scheduler_args.lr_decay_style is not None
            and current_step < lr_decay_starting_step + lr_decay_steps
        ):
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (current_step - lr_decay_starting_step)
                            / lr_decay_steps
                        )
                    )
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (lr_decay_steps - (current_step - lr_decay_starting_step))
                    / lr_decay_steps
                )
            elif lr_scheduler_args.lr_decay_style == "1-sqrt":
                lmbda = lr_scheduler_args.min_decay_lr + (
                    initial_lr - lr_scheduler_args.min_decay_lr
                ) * (
                    1
                    - math.sqrt(
                        (current_step - lr_decay_starting_step) / lr_decay_steps
                    )
                )
            else:
                raise ValueError(
                    f"Unknown decay style {lr_scheduler_args.lr_decay_style}"
                )

        # Optional constant phase at min_decay_lr
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= initial_lr  # Normalization for pytorch
        return lmbda

    def get_lr_lambda_for_param_group(lr: float):
        return partial(lr_lambda, initial_lr=lr)

    # NOTE: get learning rate scheduler for each param group
    # NOTE: Changes made.
    lr_lambdas = []
    for param_group in optimizer.param_groups:
        lr_lambdas.append(get_lr_lambda_for_param_group(lr=param_group["lr"]))

    assert len(lr_lambdas) == len(optimizer.param_groups), (
        "Custom learning rate functions dont match the number of param groups"
    )

    log_rank(
        f"[Optimizer Building] There are total {len(lr_lambdas)} custom learning rate function for parameter groups",
        logger=logger,
        level=logging.DEBUG,
    )

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)
    return lr_scheduler
