#!/bin/bash
#################### 环境变量 ####################
export CUDA_VISIBLE_DEVICES="0,1"
export HF_HOME="~/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    torchrun --nproc_per_node=1 --master_port 25675 \
        -m src.conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name ~/data/models/HuggingFaceTB/SmolLM-135M \
        --auto_encoder

    accelerate launch --num_processes=${NUM_GPUS} --main_process_port 25675 \
        -m src.auto_encoder.eval --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 96 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
        --output_dir "../eval_results/${output_dir}"
}

eval_all() {
    local model_name_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

    # eval所有检查点
    matching_directories=$(find "$model_name_path" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+')

    echo $matching_directories

    for dir in $matching_directories; do
        echo "Evaluating $dir"
        eval_one_ckpt $dir $output_dir $cfg_RoPE
    done
}

#################### 任务执行 ####################
set -e

eval_one_ckpt ../checkpoints/${MODEL_NAME}/18000 "${MODEL_NAME}" ../configs/ae/${MODEL_NAME}.yaml