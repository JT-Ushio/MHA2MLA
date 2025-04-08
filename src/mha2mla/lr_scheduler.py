import math
import logging
from nanotron.logging import log_rank
from functools import partial
from torch.optim import Optimizer
from types import SimpleNamespace
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


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
    lr_lambdas = []
    for param_group in optimizer.param_groups:
        lr_lambdas.append(get_lr_lambda_for_param_group(lr=param_group["lr"]))

    assert len(lr_lambdas) == len(
        optimizer.param_groups
    ), "Custom learning rate functions dont match the number of param groups"

    log_rank(
        f"[Optimizer Building] There are total {len(lr_lambdas)} custom learning rate function for parameter groups",
        logger=logger,
        level=logging.DEBUG,
    )

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)
    return lr_scheduler


def load_scheduler(optimizer, training_args):
    """Load learning rate scheduler from configuration."""
    # adamW
    import json

    # lr_scheduler
    lr_scheduler_kwargs = training_args.lr_scheduler_kwargs
    if isinstance(lr_scheduler_kwargs, str):
        lr_scheduler_kwargs = json.loads(lr_scheduler_kwargs)
    lr_scheduler = lr_scheduler_builder(
        optimizer=optimizer,
        lr_scheduler_args=SimpleNamespace(**lr_scheduler_kwargs),
        total_training_steps=training_args.max_steps,
    )
    return lr_scheduler
