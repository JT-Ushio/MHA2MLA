import sys
import argparse

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2ForCausalLM,
    LlamaForCausalLM,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
)
from arguments import (
    MHA2MLAModelArguments,
    MHA2MLADataArguments,
    MHA2MLATrainingArguments,
)
from helpers import load_dataset, load_optimizer_scheduler
from patching_model_load import patch_model
from patching_qwen2 import mha2mla_qwen2
from patching_llama import mha2mla_llama


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
    if resume_from_checkpoint is None:
        mha_model = AutoModelForCausalLM.from_pretrained(name)
        mla_model, q_idx, k_idx = patch_model(mha_model, model_args, mha2mla_args)
    else:
        mha_model = AutoModelForCausalLM.from_config(model_args)
        mla_model, q_idx, k_idx = patch_model(mha_model, model_args, mha2mla_args)
        mla_state_dict = AutoModelForCausalLM.from_pretrained(resume_from_checkpoint).state_dict()
        mla_model.load_state_dict(mla_state_dict)

    # print args and patch mha2mla
    print(model_args, mha2mla_args, mha_model)
    if isinstance(mha_model, LlamaForCausalLM):
        mha2mla_llama(q_idx, k_idx)
    elif isinstance(mha_model, Qwen2ForCausalLM):
        mha2mla_qwen2(q_idx, k_idx)
    print(mla_model)

    if train_args.bf16:
        mla_model = mla_model.to(dtype=torch.bfloat16)
    elif train_args.fp16:
        mla_model = mla_model.to(dtype=torch.float16)

    train_dataset = load_dataset(data_args, train_args, tokenizer)
    resume_from_checkpoint = train_args.resume_from_checkpoint
    optimizer, lr_scheduler = load_optimizer_scheduler(mla_model, train_args)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=mla_model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator,
    )
    # train
    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
