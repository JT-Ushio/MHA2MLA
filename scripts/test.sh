#################### 环境变量 ####################

export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="~/data/hf-home"
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
export MASTER_PORT="auto"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export model_name_or_path="/home/binguo/data/models/HuggingFaceTB/SmolLM-135M"
export PYTHONPATH=..:$PYTHONPATH
export PYTHONPATH=../mha2mla:$PYTHONPATH

#################### 任务执行 ####################

set -e

# torchrun --nproc_per_node 1 \
#     ../src/mha2mla_nt/2_norm.py \
#     --config-file ../configs/test/135M_2norm.yaml \
#     --output-dir ../utils/ \
#     --sample-size 1024

# torchrun --nproc_per_node 1 --master_port 24575 \
#     ../src/mha2mla/run_train.py \
#     --config_file ../configs_hf/rope/135M_2GPU_mla.yaml \
#     --partial_rope_config ../configs_hf/rope/v4_topk4_rope.yaml


# torchrun --nproc_per_node 1 --master_port 24575 \
#     ../src/mha2mla/2_norm.py \
#     --config_file ../configs_hf/rope/135M_2GPU.yaml \
#     --output_dir ./qk_tensor_hf_test.pth \
#     --sample_size 1024

# torchrun --nproc_per_node 1 --master_port 24558 \
#     -m src.test.inference \
#     --model_name /home/binguo/data/MLA-FT/checkpoints/1.7B_rope_v4_topk4_svd_method7_rank32 \
#     --is_mla

# torchrun --nproc_per_node 1 --master_port 24558 \
#     -m src.test.inference \
#     --model_name /home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B

# compare the loss diff between the infer mode and the train mode

# torchrun --nproc_per_node 1 --master_port 24558 \
#     ../test_dbg/test/inference.py \
#     --model_name /home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank32/18000_hf \
#     --is_mla

# torchrun --nproc_per_node 1 --master_port 24558 \
#     ../test_dbg/test/inference.py \
#     --model_name /home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank32/18000_hf \
#     --is_mla \
#     --is_inference


# torchrun --nproc_per_node 1 --master_port 24558 \
#     ../test_dbg/test/inference.py \
#     --model_name /home/binguo/data/MLA-FT/checkpoints/rope_v4_topk4_svd_method7_rank32/18000_hf \
#     --is_mla \
#     --is_inference

