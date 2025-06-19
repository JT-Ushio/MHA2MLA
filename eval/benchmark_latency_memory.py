import os
import argparse
from time import perf_counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from torch.profiler import record_function

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, QuantizedCacheConfig,QuantizedCache

from deepspeed.profiling.flops_profiler import FlopsProfiler

@torch.no_grad()
def prefill(model, inputs, cache_implementation, cache_kwargs:dict|str):
    if cache_implementation == "quantized":
        if isinstance(cache_kwargs, str):
            import json
            cache_kwargs = json.loads(cache_kwargs)
        cache_config = {
            "backend": cache_kwargs["backend"],
            "nbits": cache_kwargs["nbits"],
            "compute_dtype": model.dtype,
            "device": model.device,
        }
        cache_config = QuantizedCacheConfig(**cache_config)
        past_key_values = QuantizedCache(cache_config)
    else:
        past_key_values = DynamicCache()

    input_length = inputs["input_ids"].shape[1]
    inputs["cache_position"] = torch.arange(input_length, device=inputs["input_ids"].device)
    outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    next_input_ids = torch.cat([inputs["input_ids"], next_tokens[:, None]], dim=-1)
    next_model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            inputs,
            is_encoder_decoder=False,
        )
    return next_input_ids, next_model_kwargs

def generate_random_input_ids(batch_size, input_length, vocab_size, device):
    return torch.randint(0, vocab_size, (batch_size, input_length), device=device)

def plot_comparison(df, feature, metric, output_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=feature, y=metric, hue='model')
    plt.title(f"{metric} vs {feature}")
    plt.xlabel(feature)
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_{feature}_comparison.png"))
    plt.close()

@torch.no_grad()
def eval_generated_lengths(model, tokenizer, cache_implementation, nbits, feature, model_name, results):
    generate_kwargs = {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
    parameters = {"max_new_tokens": 1024, "batch_size": 16, "input_length": 128}
    num_batches = 1

    # warm up
    generate_kwargs = {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
    for _ in range(3):
        inputs_warmup = tokenizer(["Today a dragon flew over Paris"] * 16, return_tensors="pt").to(model.device)
        model.generate(**inputs_warmup, max_new_tokens=1024, **generate_kwargs)

    if feature == "batch_size":
        x_iterable = [1, 4, 16, 64, 256]
    else:
        x_iterable = [512, 1024, 2048]

    vocab_size = tokenizer.vocab_size

    for item in x_iterable:
        parameters[feature] = item
        generate_kwargs_curr = generate_kwargs.copy()
        generate_kwargs_curr["min_new_tokens"] = parameters["max_new_tokens"]
        generate_kwargs_curr["max_new_tokens"] = parameters["max_new_tokens"]

        batch_size = parameters["batch_size"]
        flops = []
        duration = []
        memory = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 使用 PyTorch Profiler 统计显存和时间
        for batch in range(num_batches):
            start = perf_counter()
            input_ids = generate_random_input_ids(
                batch_size, 
                parameters["input_length"], 
                vocab_size, 
                model.device
            )
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)
            }
            
            # pre-fill stage
            with record_function("prefill"):
                next_input_ids, next_model_kwargs = prefill(model, inputs, cache_implementation, nbits)
                ttft = perf_counter() - start
                next_model_kwargs.pop("input_ids")

            profiler = FlopsProfiler(model)
            profiler.start_profile()

            # decoding stage
            with record_function("decoding"):
                out = model.generate(
                    next_input_ids,
                    **next_model_kwargs,
                    **generate_kwargs_curr
                )
                decoding_time = (perf_counter() - start - ttft) / batch_size / parameters["max_new_tokens"]

            del out
            torch.cuda.empty_cache()

            profiler.stop_profile()
            flops.append(profiler.get_total_flops(as_string=False))
            duration.append(profiler.get_total_duration(as_string=False) / parameters["max_new_tokens"])
            profiler.end_profile()

            memory.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        tokens_per_sec = 1 / decoding_time
        mean_flops = sum(flops) / len(flops)
        mean_duration = sum(duration) / len(duration)
        mean_memory = sum(memory) / len(memory)

        print(f"Model: {model_name}, {feature}: {item}, Flops:{mean_flops}, Duration:{mean_duration}, Memory:{mean_memory}, Tokens per second: {tokens_per_sec}, Time to first token: {ttft}")
        results.append({
            'model': model_name,
            feature: item,
            'memory': mean_memory,
            'flops': mean_flops,
            'duration': mean_duration,
            'tokens_per_second': tokens_per_sec,
            'time_to_first_token': ttft
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,required=True)
    parser.add_argument("--cache_implementation", type=str, default="dynamic")
    parser.add_argument("--cache_kwargs", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--is_mla", action="store_true")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.is_mla:
        from monkey_patch import infer_monkey_patch
        import json
        with open(os.path.join(args.model_path, "config.json")) as f:
            config = json.load(f)
        infer_monkey_patch(config["RoPE"])

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    results = []

    model_path = args.model_path
    print(f"Evaluating model: {model_path}")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True, 
        torch_dtype=dtype,
    ).to("cuda:0")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    for feature in ["batch_size", "max_new_tokens"]:
        eval_generated_lengths(
            model,
            tokenizer,
            cache_implementation=args.cache_implementation,
            nbits=args.cache_kwargs,
            feature=feature,
            model_name=model_path,
            results=results
        )

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_path), index=False)


def save_dual_subplots(title, x1, y1_list, xlabel1, x2, y2_list, xlabel2, ylabel, output_path, labels=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    width = 0.35  # Width of the bars
    x1_pos = np.arange(len(x1))
    x2_pos = np.arange(len(x2))
    
    for i, y in enumerate(y1_list):
        offset = width * (i - len(y1_list)/2 + 0.5)
        axes[0].bar(x1_pos + offset, y, width, label=labels[i] if labels else f'Model {i+1}')
    axes[0].set_title(f"{title} - {xlabel1}")
    axes[0].set_xticks(x1_pos)
    axes[0].set_xticklabels(x1)
    axes[0].set_xlabel(xlabel1)
    axes[0].set_ylabel(ylabel)
    if labels:
        axes[0].legend()
    
    for i, y in enumerate(y2_list):
        offset = width * (i - len(y2_list)/2 + 0.5)
        axes[1].bar(x2_pos + offset, y, width, label=labels[i] if labels else f'Model {i+1}')
    axes[1].set_title(f"{title} - {xlabel2}")
    axes[1].set_xticks(x2_pos)
    axes[1].set_xticklabels(x2)
    axes[1].set_xlabel(xlabel2)
    axes[1].set_ylabel(ylabel)
    if labels:
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

def plot():
    file_path = [
        # "../eval_results/cost/135M_bf16.csv",
        "../eval_results/cost/135M_rope_v4_topk4_svd_method7_rank16_A_CC_ME.csv",
        "../eval_results/cost/135M_rope_v4_topk4_svd_method7_rank16_AM_CC_ME.csv",
        "../eval_results/cost/135M_rope_v4_topk4_svd_method7_rank16.csv",
        # "../eval_results/cost/135M_rope_v4_topk4_svd_method7_rank32_AM_CC_ME.csv",
        # "../eval_results/cost/135M_rope_v4_topk4_svd_method7_rank32.csv",
    ]
    labels = [
        # "135M_baseline",
        "135M_rope_v4_topk4_svd_method7_rank16_A_CC_ME",
        "135M_rope_v4_topk4_svd_method7_rank16_AM_CC_ME",
        "135M_rope_v4_topk4_svd_method7_rank16",
        # "135M_rope_v4_topk4_svd_method7_rank32_AM_CC_ME",
        # "135M_rope_v4_topk4_svd_method7_rank32",
    ]
    output_dir = "/home/binguo/data/MLA-FT/eval_results/cost"
    
    memory_usage_batch = [[] for _ in labels]
    flops_usage_batch = [[] for _ in labels]
    duration_usage_batch = [[] for _ in labels]
    memory_usage_tokens = [[] for _ in labels]
    flops_usage_tokens = [[] for _ in labels]
    duration_usage_tokens = [[] for _ in labels]

    
    for idx, f in enumerate(file_path):
        df = pd.read_csv(f)
        
        batch_df = df.dropna(subset=["batch_size"])
        batch_values = batch_df["batch_size"].values.tolist()
        memory_usage_batch[idx] = batch_df["memory"].values.tolist()
        flops_usage_batch[idx] = batch_df["flops"].values.tolist()
        duration_usage_batch[idx] = (batch_df["duration"]).values.tolist()
        
        tokens_df = df.dropna(subset=["max_new_tokens"])
        tokens_values = tokens_df["max_new_tokens"].values.tolist()
        memory_usage_tokens[idx] = tokens_df["memory"].values.tolist()
        flops_usage_tokens[idx] = tokens_df["flops"].values.tolist()
        duration_usage_tokens[idx] = (tokens_df["duration"]).values.tolist()

    
    save_dual_subplots(
        "GPU Memory Consumption",
        batch_values,
        memory_usage_batch,
        "Batch Size",
        tokens_values,
        memory_usage_tokens,
        "Max New Tokens",
        "Memory (MB)",
        f"{output_dir}/memory_combined.png",
        labels=labels
    )
    
    save_dual_subplots(
        "Flops Usage",
        batch_values,
        flops_usage_batch,
        "Batch Size",
        tokens_values,
        flops_usage_tokens,
        "Max New Tokens",
        "Flops",
        f"{output_dir}/flops_combined.png",
        labels=labels
    )

    save_dual_subplots(
        "Duration",
        batch_values,
        duration_usage_batch,
        "Batch Size",
        tokens_values,
        duration_usage_tokens,
        "Max New Tokens",
        "Duration (s)",
        f"{output_dir}/duration_combined.png",
        labels=labels
    )

if __name__ == "__main__":
    plot()
    # main()
