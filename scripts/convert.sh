#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH

#################### 任务执行 ####################

model_name_or_path=/home/binguo/data/models/meta-llama/Llama-2-7b-hf

torchrun --nproc_per_node=1 --master_port 25675 \
    -m src.conversation.convert_hf_to_nanotron \
    --checkpoint_path ${model_name_or_path} \
    --save_path "${model_name_or_path}_nt" \
