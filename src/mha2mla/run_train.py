import torch
import os
import sys
import argparse

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
)
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2SdpaAttention,
    Qwen2RotaryEmbedding,
)

from arguments import (
    MHA2MLAModelArguments,
    MHA2MLADataArguments,
    MHA2MLATrainingArguments,
)
from helpers import load_dataset, load_optimizer_scheduler
from patching_model_load import patch_model
from patching_qwen2 import (
    mha2mla_qwen2
    # custom_Qwen2SdpaAttention_forward,
    # custom_Qwen2RotaryEmbedding_forward,
    # create_custom_apply_rotary_pos_emb,
)


def main():
    cfg_parser = argparse.ArgumentParser()
    cfg_parser.add_argument("--cfg_file", type=str, required=True)
    cfg = cfg_parser.parse_args()
    hf_parser = HfArgumentParser(
        (MHA2MLATrainingArguments, MHA2MLAModelArguments, MHA2MLADataArguments)
    )
    train_args, mha2mla_args, data_args = hf_parser.parse_yaml_file(cfg.cfg_file)
    name = mha2mla_args.model_name_or_path
    model_args = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    mha_model = AutoModelForCausalLM.from_pretrained(name)
    mla_model, q_idx, k_idx = patch_model(mha_model, model_args, mha2mla_args)
    mha2mla_qwen2(q_idx, k_idx)
    # Qwen2SdpaAttention.forward = custom_Qwen2SdpaAttention_forward
    # Qwen2RotaryEmbedding.forward = custom_Qwen2RotaryEmbedding_forward
    # modeling_qwen2.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb(
    #     q_idx, k_idx
    # )
    print(model_args, mha2mla_args)
    print(mha_model, mla_model)

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
        # if int(os.getenv("LOCAL_RANK", 0)) == 0 and train_args.save_initial_model:
        #     trainer._save_checkpoint()
        trainer.train()


if __name__ == "__main__":
    main()
