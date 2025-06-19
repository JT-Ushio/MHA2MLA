import torch
from transformers import (
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Qwen2ForCausalLM,
    LlamaForCausalLM,
)
import os
from tqdm import tqdm
import datasets
import numpy as np

from nanotron.data.nanoset import Nanoset
from arguments import (
    MHA2MLADataArguments,
    QKNormArguments,
)


def load_dataset(dataset_args, qknorm_args, tokenizer):
    """Load dataset from configuration."""
    tokenizer.model_max_length = dataset_args.sequence_length
    if dataset_args.is_nanoset:
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        dataset = Nanoset(
            dataset_folders=dataset_args.dataset_folders,
            sequence_length=dataset_args.sequence_length,
            dataset_weights=dataset_args.dataset_weights,
            token_size=token_size,
            train_split_num_samples=qknorm_args.sample_size,
        )
    else:
        import datasets

        dataset = datasets.load_dataset(
            dataset_args.dataset_name_or_path, split="train"
        )

    return dataset


def load_tokenizer_and_model(qknorm_args: QKNormArguments):
    """Load tokenizer and model from configuration."""
    assert qknorm_args.model_name_or_path is not None, (
        "Must provide the path to the model"
    )
    qknorm_args.tokenizer_name_or_path = qknorm_args.model_name_or_path

    config = AutoConfig.from_pretrained(qknorm_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        qknorm_args.model_name_or_path, config=config
    )

    tokenizer = AutoTokenizer.from_pretrained(qknorm_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


hidden_states_dict = {}


def create_hook_fn(name):
    def hook(module, args, kwargs, output):
        hidden_states_dict[name] = kwargs["hidden_states"]

    return hook


def main():
    import argparse

    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = cmd_parser.parse_args()
    hf_parser = HfArgumentParser((
        MHA2MLADataArguments,
        QKNormArguments,
    ))
    data_args, qknorm_args = hf_parser.parse_yaml_file(args.config_file)

    model, tokenizer = load_tokenizer_and_model(qknorm_args)
    train_dataset = load_dataset(data_args, qknorm_args, tokenizer)
    assert int(os.getenv("WORLD_SIZE", 1)) == 1, "Only support single process."

    def preprocess_function(examples):
        if "input_ids" in examples:
            return {"input_ids": examples["input_ids"]}
        elif "text" in examples:
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            raise ValueError(
                "Unsupported dataset format. Must be a dictionary containing 'input_ids' or 'text'."
            )

    if isinstance(train_dataset, datasets.Dataset):
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids"])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    batch_size = qknorm_args.batch_size
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=data_collator,
    )
    data_iter = iter(data_loader)
    num = qknorm_args.sample_size
    model.eval()
    model.to("cuda")

    if isinstance(model, Qwen2ForCausalLM):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        attn = Qwen2Attention
    elif isinstance(model, LlamaForCausalLM):
        from transformers.models.llama.modeling_llama import LlamaAttention

        attn = LlamaAttention

    for name, module in model.named_modules():
        if not isinstance(module, attn):
            continue
        hook_fn = create_hook_fn(
            name
        )  # name:'model.layers.0.self_attn'moudle:LlamaSdpaAttention
        module.register_forward_hook(hook_fn, with_kwargs=True)

    p_bar = tqdm(total=num)
    model_config = model.config
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_layers = model_config.num_hidden_layers
    query_states = [[] for _ in range(num_layers)]
    key_states = [[] for _ in range(num_layers)]

    def cal_2_norm(states):
        states = torch.norm(
            states.reshape(
                states.shape[0], states.shape[1], states.shape[2], 2, -1
            ).transpose(-1, -2),
            p=2,
            dim=4,
        )
        return states

    with torch.no_grad():
        for _ in range(0, num, batch_size):
            batch = next(data_iter)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            model(**batch)
            p_bar.update(batch_size)
            for name, module in model.named_modules():
                if not isinstance(module, attn):
                    continue
                idx = int(name.split(".")[2])
                bsz, q_len, _ = hidden_states_dict[name].shape
                q = module.q_proj(hidden_states_dict[name]).reshape(
                    bsz, q_len, model_config.num_attention_heads, head_dim
                )  # [bsz,q_len,num_heads,head_dim]
                k = module.k_proj(hidden_states_dict[name]).reshape(
                    bsz, q_len, model_config.num_key_value_heads, head_dim
                )
                query_states[idx].append(
                    cal_2_norm(q).mean(dim=1, keepdim=False).cpu()
                )  # [bsz,num_heads,head_dim//2]
                key_states[idx].append(cal_2_norm(k).mean(dim=1, keepdim=False).cpu())

    query_states = torch.stack(
        [torch.cat(query_states[i], dim=0) for i in range(num_layers)], dim=0
    )  # [num_layers,sample_size,num_heads,head_dim//2]
    key_states = torch.stack(
        [torch.cat(key_states[i], dim=0) for i in range(num_layers)], dim=0
    )
    query_states = torch.mean(
        query_states, dim=1, keepdim=False
    )  # [num_layers,num_heads,head_dim//2]
    key_states = torch.mean(key_states, dim=1, keepdim=False)
    group_size = model_config.num_attention_heads // model_config.num_key_value_heads
    key_states = (
        key_states.unsqueeze(2)
        .expand(
            num_layers,
            model_config.num_key_value_heads,
            group_size,
            head_dim // 2,
        )
        .reshape(num_layers, model_config.num_attention_heads, head_dim // 2)
    )  # [num_layers,num_heads,head_dim//2]
    qk_states = query_states * key_states
    if group_size > 1:
        qk_states = qk_states.reshape(
            num_layers, model_config.num_key_value_heads, group_size, head_dim // 2
        ).sum(dim=2, keepdim=False)
    _, sorted_indices = torch.sort(qk_states, dim=-1, descending=True)
    ranks = torch.empty_like(sorted_indices, dtype=torch.uint8)  # ,dtype=torch.uint8
    rank_values = torch.arange(qk_states.shape[-1], dtype=torch.uint8).expand_as(
        qk_states
    )
    ranks.scatter_(-1, sorted_indices, rank_values)
    ranks = torch.cat([ranks, ranks], dim=-1)
    with open(qknorm_args.qk_output_dir, "wb") as f:
        torch.save(ranks, f)


if __name__ == "__main__":
    main()
