#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="1"
export HF_HOME="~/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export PYTHONPATH=..:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

#################### 任务执行 ####################

# python ../src/mha2mla/benchmark_latency_memory.py \
#     --model_path /home/binguo/data/models/HuggingFaceTB/SmolLM-135M \
#     --dtype bf16 \
#     --cache_implementation dynamic \
#     --output_path ../eval_results/cost/135M_bf16.csv

# python ../src/mha2mla/benchmark_latency_memory.py \
#     --model_path /home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank16/18000_hf \
#     --dtype bf16 \
#     --cache_implementation dynamic \
#     --output_path ../eval_results/cost/135M_rope_v4_topk4_svd_method7_rank16.csv \
#     --is_mla

python ../src/mha2mla/benchmark_latency_memory.py
