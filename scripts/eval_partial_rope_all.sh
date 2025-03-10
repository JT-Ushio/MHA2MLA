#!/bin/bash
#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0,1,2,3"
<<<<<<< HEAD
export HF_HOME="~/data/hf-home"
=======
>>>>>>> feature/low-rank-approx
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH

#################### 函数定义 ####################

eval_one_ckpt() {
    local model_name_or_path=$1
    local output_dir=$2
    local cfg_RoPE=$3

<<<<<<< HEAD
    torchrun --nproc_per_node=1 \
        -m src.original_conversation.convert_nanotron_to_hf \
        --checkpoint_path ${model_name_or_path} \
        --save_path "${model_name_or_path}_hf" \
        --tokenizer_name ~/data/models/HuggingFaceTB/SmolLM-135M

    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
        -m src.evaluation.eval_partial_rope --cfg_RoPE ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path}_hf,revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 96 \
        --custom_tasks "../src/evaluation/tasks.py" \
        --tasks "../src/evaluation/smollm1_base_v2.txt" \
=======
    # torchrun --nproc_per_node=1 \
    #     -m src.original_conversation.convert_nanotron_to_hf \
    #     --checkpoint_path ${model_name_or_path} \
    #     --save_path "${model_name_or_path}_hf" \
    #     --tokenizer_name ~/data/models/HuggingFaceTB/SmolLM-135M

    accelerate launch --multi_gpu --num_processes=${NUM_GPUS} \
        ../src/mha2mla/eval.py --partial_rope_config ${cfg_RoPE} \
        accelerate \
        --model_args "pretrained=${model_name_or_path},revision=main,dtype=bfloat16,max_length=2048" \
        --override_batch_size 48 \
        --custom_tasks "../src/mha2mla/tasks.py" \
        --tasks "../src/mha2mla/smollm1_base.txt" \
>>>>>>> feature/low-rank-approx
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

<<<<<<< HEAD
eval_one_ckpt ../checkpoints/v1_2_rope/18000 v1_2_rope ../configs/rope/v1_2_rope.yaml
=======
eval_one_ckpt /cpfs01/shared/llm_ddd/doushihan/taoji/code/MLA-FT/checkpoints/test_lr4e-5/checkpoint-18000 hf_test_lr4e-5 ../configs_hf/rope/rope_v4_topk4.yaml
>>>>>>> feature/low-rank-approx
