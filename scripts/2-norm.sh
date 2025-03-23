#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=../:$PYTHONPATH
# export MASTER_PORT="auto"

torchrun --nproc_per_node 1 \
   ../src/mha2mla_nt/2_norm.py \
   --config-file ../configs/test/7B_2norm.yaml \
   --output-dir . \
   --sample-size 1024
