## Init AutoEncoder Module
Prepare configuration file referencing [init.yaml](../../configs/ae/init.yaml).

Design the AutoEncoder module by configuring the RoPE and auto_encoder parameters.

Init command:
```bash
torchrun --nproc_per_node ${N_GPU} \
    src/auto_encoder/init.py \
    --config_file configs/ae/init.yaml
```

## Train Model with AutoEncoder Module
Prepare general fine-tune configuration file referencing [test.yaml](../../configs/ae/test.yaml).

Set the `model_name_or_path` and `tokenizer_name_or_path` parameter to the checkpoint of init.

FT command:
```bash
torchrun --nproc_per_node ${N_GPU} \
    src/auto_encoder/train.py \
    --config_file configs/ae/test.yaml \
    --is_auto_encoder
```

## Eval

Eval command:
```bash
export MODEL_PATH="./checkpoints/ae_test/checkpoint-12000"
accelerate launch --num_processes=${N_GPU} \
    src/auto_encoder/eval.py --is_auto_encoder \
    accelerate \
    --model_args "pretrained=${MODEL_PATH},revision=main,dtype=bfloat16,max_length=2048" \
    --override_batch_size 200 \
    --custom_tasks "src/mha2mla/tasks.py" \
    --tasks "src/mha2mla/smollm1_base.txt" \
    --output_dir "eval_results/ae_test"
```
