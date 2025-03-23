#!/bin/bash
set -e
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH=..:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_PORT="auto"
export WANDB_PROJECT="mla_smollm_ft_nt"

#################### 任务执行 ####################


torchrun --nproc_per_node 1 \
    -m src.auto_encoder.init \
    --config-file ../configs/ae/init_hf.yaml

# torchrun --nproc_per_node 1 --master_port 24558 \
#     -m src.auto_encoder.merge \
#     --original_model_path ../checkpoints/HuggingFaceTB/SmolLM-1.7B_nt \
#     --ae_model_path ../checkpoints/1.7B_rope_v4_topk4_ae_v2_rank8_null_init/checkpoint-2000 \
#     --save_path ../checkpoints/1.7B_rope_v4_topk4_ae_v2_rank8_null/0

# torchrun --nproc_per_node 8 --master_port 24558 \
#     -m src.auto_encoder.train \
#     --config-file ../configs/ae/1.7B_rope_v4_topk4_ae_v2_rank8_null.yaml
