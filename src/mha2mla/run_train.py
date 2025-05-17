import argparse
from dataclasses import asdict

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    LlamaForCausalLM,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
)
from transformers.utils import logging
from arguments import (
    MHA2MLAModelArguments,
    MHA2MLADataArguments,
    MHA2MLATrainingArguments,
)
from helpers import load_dataset, load_optimizer_scheduler
from patching_model_load import patch_model
from patching_qwen2 import mha2mla_qwen2
from patching_qwen3 import mha2mla_qwen3
from patching_llama import mha2mla_llama

logger = logging.get_logger(__name__)


def main():
    # load arguments
    cfg_parser = argparse.ArgumentParser()
    cfg_parser.add_argument("--cfg_file", type=str, required=True)
    cfg = cfg_parser.parse_args()
    hf_parser = HfArgumentParser(
        (MHA2MLATrainingArguments, MHA2MLAModelArguments, MHA2MLADataArguments)
    )
    train_args, mha2mla_args, data_args = hf_parser.parse_yaml_file(cfg.cfg_file)

    # load tokenizer and model
    name = mha2mla_args.model_name_or_path
    model_args = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resume_from_checkpoint = train_args.resume_from_checkpoint
    if mha2mla_args.is_baseline or resume_from_checkpoint is None:
        mha_model = AutoModelForCausalLM.from_pretrained(name)
    else:
        mha_model = AutoModelForCausalLM.from_config(model_args)
    if not mha2mla_args.is_baseline or mha2mla_args.is_mla_from_scratch:
        mla_model, q_idx, k_idx = patch_model(mha_model, model_args, mha2mla_args)
        if isinstance(mha_model, LlamaForCausalLM):
            mha2mla_llama(q_idx, k_idx)
        elif isinstance(mha_model, Qwen2ForCausalLM):
            mha2mla_qwen2(q_idx, k_idx)
        elif isinstance(mha_model, Qwen3ForCausalLM):
            mha2mla_qwen3(q_idx, k_idx)
    model = mha_model if mha2mla_args.is_baseline else mla_model
    model.config.mha2mla = asdict(mha2mla_args)

    if train_args.bf16:
        model = model.to(dtype=torch.bfloat16)
    elif train_args.fp16:
        model = model.to(dtype=torch.float16)

    train_dataset = load_dataset(data_args, train_args, tokenizer)
    optimizer, lr_scheduler = load_optimizer_scheduler(model, train_args)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint)
    trainer.log(asdict(mha2mla_args))


if __name__ == "__main__":
    main()
