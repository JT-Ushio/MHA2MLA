import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    DataCollatorForLanguageModeling,
)
from nanotron.data.nanoset import Nanoset
from nanotron.parallel.pipeline_parallel.block import TensorPointer
import os
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import yaml
from accelerate import __version__ as accelerate_version
from packaging import version
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.cache_utils import DynamicCache, Cache
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)
import functools

from lr_scheduler import load_scheduler as load_scheduler4constant_with_warmup_decay


@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_constant_with_warmup_decay_scheduler: bool = False
    compute_loss: str = "v0"


@dataclass
class ModelArguments:
    model_name_or_path: str = None
    tokenizer_name_or_path: str = None
    save_initial_model: bool = False
    RoPE: dict = None
    auto_encoder: dict = None


@dataclass
class DataArguments:
    is_nanoset: bool = False
    dataset_folders: List[str] = None
    dataset_weights: List[float] = None
    dataset_name_or_path: str = None
    sequence_length: int = 2048


TYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def process_state_dict_for_llama(model, state_dict):
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_idx}.self_attn."
        Q = state_dict.pop(prefix + "q_proj.weight")
        K = state_dict.pop(prefix + "k_proj.weight")
        nope_mask_for_q = layer.self_attn.nope_mask_for_q
        nope_mask_for_k = layer.self_attn.nope_mask_for_k
        state_dict[prefix + "W_q_rope.weight"] = Q[~nope_mask_for_q, :]
        state_dict[prefix + "W_q_nope.weight"] = Q[nope_mask_for_q, :]
        state_dict[prefix + "W_k_rope.weight"] = K[~nope_mask_for_k, :]
        state_dict[prefix + "W_k_nope.weight"] = K[nope_mask_for_k, :]
    return state_dict


CLASS2FUNC = {
    "LlamaForCausalLM": process_state_dict_for_llama,
}


def get_key_value_for_llama(attn, inputs):
    hidden_states = inputs[1]["hidden_states"]
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)
    return key_states, value_states


CLASS2KVFUNC = {
    "LlamaForCausalLM": get_key_value_for_llama,
}


class AutoEncoderInitTrainer(Trainer):
    @functools.wraps(Trainer.__init__)
    def __init__(
        self,
        **kwargs,
    ):
        original_model = kwargs.pop("model").cuda()
        self.original_model = original_model
        self.loss_func = torch.nn.SmoothL1Loss(reduction="sum")
        # Instance AutoEncoder Model
        from monkey_patch import ae_patch_func_hf

        ae_patch_func_hf()
        model = original_model.__class__(original_model.config).cuda()
        # Split model state_dict
        original_state_dict = original_model.state_dict()
        original_state_dict = CLASS2FUNC[original_model.__class__.__name__](
            model, original_state_dict
        )
        # Initialize auto encoder
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if "auto_encoder" in key:
                if len(state_dict[key].shape) > 1:
                    # Xavier_uniform for 2D weights
                    state_dict[key] = torch.nn.init.xavier_uniform_(value)
                else:
                    # Gaussian for 1D weights
                    state_dict[key] = torch.nn.init.normal_(value)
                original_state_dict[key] = state_dict[key]
        model.load_state_dict(original_state_dict)
        # Freeze original model
        for name, named_param in original_model.named_parameters():
            named_param.requires_grad = False
        for name, named_param in model.named_parameters():
            if "auto_encoder" in name:
                named_param.requires_grad = True
            else:
                named_param.requires_grad = False
        # Monkey Patch for original k_nope and value_states
        self.inputs = [[] for _ in enumerate(self.original_model.model.layers)]

        def create_hook_fn_for_original(layer_idx):
            def hook(module, args, kwargs, output):
                self.inputs[layer_idx] = (args, kwargs)

            return hook

        for layer_idx, layer in enumerate(self.original_model.model.layers):
            attn = layer.self_attn
            attn.register_forward_hook(
                create_hook_fn_for_original(layer_idx), with_kwargs=True
            )
        self.original_kv_func = CLASS2KVFUNC[original_model.__class__.__name__]
        # Monkey Patch for ae
        self.ae_output = [[] for _ in enumerate(model.model.layers)]

        def create_hook_fn_for_ae(layer_idx):
            def hook(module, args, kwargs, output):
                self.ae_output[layer_idx] = output

            return hook

        for layer_idx, layer in enumerate(model.model.layers):
            layer.self_attn.auto_encoder.register_forward_hook(
                create_hook_fn_for_ae(layer_idx), with_kwargs=True
            )
        super().__init__(
            model=model,
            **kwargs,
        )
        self.model_accepts_loss_kwargs = False

    def compute_loss_v0(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        _ = self.original_model(
            use_cache=False,
            **inputs,
        )
        all_loss = []
        for layer_idx, layer in enumerate(self.original_model.model.layers):
            target_ae = self.model.model.layers[layer_idx].self_attn.auto_encoder
            original_key_states, original_value_states = self.original_kv_func(
                layer.self_attn, self.inputs[layer_idx]
            )
            original_k_nope = original_key_states[..., target_ae.nope_mask_for_k]
            bsz, q_len, _ = original_key_states.shape
            _, k_nope, value_states = target_ae(
                k_rope=None,
                k_nope=original_k_nope,
                value_states=original_value_states,
            )
            k_nope = k_nope.squeeze(1)
            value_states = value_states.squeeze(1)
            k_loss = self.loss_func(original_k_nope, k_nope)
            v_loss = self.loss_func(original_value_states, value_states)
            layer_loss = (k_loss + v_loss) / (bsz * q_len)
            all_loss.append(layer_loss)
        all_loss = torch.stack(all_loss, dim=0)
        all_loss = torch.mean(all_loss, dim=0, keepdim=False)
        return all_loss

    def compute_loss_v1(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        _ = self.original_model(
            use_cache=False,
            **inputs,
        )
        _ = self.model(
            use_cache=False,
            **inputs,
        )
        current_step = self.state.global_step
        max_steps = self.args.max_steps
        # assert max_steps % self.model.config.num_hidden_layers == 0
        block_step = max_steps // self.model.config.num_hidden_layers
        current_idx = current_step // block_step + 1
        all_loss = []
        for layer_idx in range(min(current_idx, self.model.config.num_hidden_layers)):
            layer = self.original_model.model.layers[layer_idx]
            target_ae = self.model.model.layers[layer_idx].self_attn.auto_encoder
            original_key_states, original_value_states = self.original_kv_func(
                layer.self_attn, self.inputs[layer_idx]
            )
            original_k_nope = original_key_states[..., target_ae.nope_mask_for_k]
            bsz, q_len, _ = original_key_states.shape
            k_nope = self.ae_output[layer_idx][1].squeeze(1)
            value_states = self.ae_output[layer_idx][2].squeeze(1)
            k_loss = self.loss_func(original_k_nope, k_nope)
            v_loss = self.loss_func(original_value_states, value_states)
            layer_loss = (k_loss + v_loss) / (bsz * q_len)
            all_loss.append(layer_loss)
        all_loss = torch.stack(all_loss, dim=0)
        all_loss = torch.mean(all_loss, dim=0, keepdim=False)
        return all_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        assert return_outputs == False, "return_outputs should be False"
        assert num_items_in_batch == None, "num_items_in_batch should be None"
        if self.args.compute_loss == "v0":
            loss = self.compute_loss_v0(
                model, inputs, return_outputs, num_items_in_batch
            )
        elif self.args.compute_loss == "v1":
            loss = self.compute_loss_v1(
                model, inputs, return_outputs, num_items_in_batch
            )
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Custom scheduler"""
        if self.lr_scheduler is None:
            if self.args.use_constant_with_warmup_decay_scheduler:
                self.lr_scheduler = load_scheduler4constant_with_warmup_decay(
                    optimizer, self.args
                )
            else:
                from transformers import get_scheduler

                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.args.max_steps,
                )
        return self.lr_scheduler


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_tokenizer_and_model(model_arguments):
    """Load tokenizer and model from configuration."""
    # model
    model_config = AutoConfig.from_pretrained(model_arguments.model_name_or_path)
    model_config.RoPE = model_arguments.RoPE
    model_config.auto_encoder = model_arguments.auto_encoder
    model_name_or_path = model_arguments.model_name_or_path
    if model_name_or_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=model_config
        )
    else:
        model = AutoModelForCausalLM(model_config)
    # tokenizer
    tokenizer_name_or_path = model_arguments.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # Warning
    return model, tokenizer


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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    hf_parser = HfArgumentParser(
        (CustomTrainingArguments, ModelArguments, DataArguments)
    )
    training_args, model_args, data_args = hf_parser.parse_dict(config)
    # Trainer
    model, tokenizer = load_tokenizer_and_model(
        model_args,
    )
    train_dataset = load_dataset(data_args, training_args, tokenizer)
    resume_from_checkpoint = training_args.resume_from_checkpoint
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    trainer = AutoEncoderInitTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
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
