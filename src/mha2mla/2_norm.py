from dataclasses import dataclass
import torch
from transformers import TrainingArguments
from transformers import HfArgumentParser, DataCollatorForLanguageModeling
import os
from tqdm import tqdm
import datasets

from run_train import (
    ModelArguments,
    DataArguments,
    load_config,
    load_dataset,
    load_tokenizer_and_model,
)

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
    cmd_parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory.",
    )
    cmd_parser.add_argument(
        "--sample_size",
        type=int,
        default=1024,
    )
    args = cmd_parser.parse_args()
    config = load_config(args.config_file)
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, dataset_args = parser.parse_dict(config)
    # assert config["DataArguments"]["DP"] == int(os.environ.get("WORLD_SIZE", 1)), "DP is not equal to WORLD_SIZE"

    # Trainer
    model, tokenizer = load_tokenizer_and_model(model_args)
    train_dataset = load_dataset(dataset_args, training_args, tokenizer)
    assert (
        int(os.getenv("WORLD_SIZE", 1)) == 1
    ), "Only support single process." 

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
            raise ValueError("Unsupported dataset format. Must be a dictionary containing 'input_ids' or 'text'.")

    if isinstance(train_dataset, datasets.Dataset):
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids"])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        drop_last=True,
        collate_fn=data_collator,
    )
    num = args.sample_size
    model.eval()
    model.to("cuda",dtype=torch.bfloat16)
    for name, module in model.named_modules():
        from transformers.models.llama.modeling_llama import LlamaAttention
        if not isinstance(module, LlamaAttention):
            continue
        hook_fn = create_hook_fn(name)
        module.register_forward_hook(hook_fn,with_kwargs=True)
    p_bar = tqdm(total=num)
    model_config = model.config
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_layers = model_config.num_hidden_layers
    query_states = [[] for _ in range(num_layers)]
    key_states = [[] for _ in range(num_layers)]
    def cal_2_norm(states):
        states = torch.norm(
            states.reshape(states.shape[0],states.shape[1],states.shape[2],2,-1).transpose(-1,-2),
            p=2,
            dim=4,
        )
        return states
    with torch.no_grad():
        for _,batch in enumerate(data_loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            model(**batch)
            num -= batch["input_ids"].shape[0]
            p_bar.update(batch["input_ids"].shape[0])
            for name,module in model.named_modules():
                if not isinstance(module, LlamaAttention):
                    continue
                idx = int(name.split(".")[2])
                bsz,q_len,_ = hidden_states_dict[name].shape
                q = module.q_proj(hidden_states_dict[name]).reshape(bsz,q_len,model_config.num_attention_heads,head_dim) # [bsz,q_len,num_heads,head_dim]
                k = module.k_proj(hidden_states_dict[name]).reshape(bsz,q_len,model_config.num_key_value_heads,head_dim)
                query_states[idx].append(cal_2_norm(q).mean(dim=1,keepdim=False).cpu()) # [bsz,num_heads,head_dim//2]
                key_states[idx].append(cal_2_norm(k).mean(dim=1,keepdim=False).cpu())
            if num <= 0:
                break
    query_states = torch.stack([torch.cat(query_states[i],dim=0) for i in range(num_layers)],dim=0) # [num_layers,sample_size,num_heads,head_dim//2]
    key_states = torch.stack([torch.cat(key_states[i],dim=0) for i in range(num_layers)],dim=0)
    query_states = torch.mean(query_states,dim=1,keepdim=False) # [num_layers,num_heads,head_dim//2]
    key_states = torch.mean(key_states,dim=1,keepdim=False)
    group_size = model_config.num_attention_heads // model_config.num_key_value_heads
    key_states = key_states.unsqueeze(2).expand(num_layers,model_config.num_key_value_heads,group_size,model_config.head_dim//2).reshape(num_layers,model_config.num_attention_heads,head_dim//2) # [num_layers,num_heads,head_dim//2]
    qk_states = query_states * key_states
    if group_size > 1:
        qk_states = qk_states.reshape(num_layers,model_config.num_key_value_heads,group_size,head_dim//2).sum(dim=2,keepdim=False)
    with open(os.path.join(args.output_dir,"qk_tensor.pth"),"wb") as f:
        torch.save(qk_states,f)

if __name__ == "__main__":
    main()
