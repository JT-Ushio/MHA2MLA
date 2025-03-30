# Nanotron

## Format conversation

If you want to convert hf ckpt to nanotron ckpt, use the command below:

```bash
torchrun --nproc_per_node=1 \
    -m src.conversation.convert_hf_to_nanotron \
    --checkpoint_path ${model_path} \
    --save_path "${model_path}_nt"
```

If you want to convert nanotron ckpt to hf ckpt, use the command below:

```bash
torchrun --nproc_per_node=1 \
    -m src.conversation.convert_nanotron_to_hf \
    --checkpoint_path ${model_path} \
    --tokenizer_name ${tokenizer_path} \
    --save_path "${model_path}_hf"
```

For the MLA model, you should add the parameter `--is_mla` to the command.

## MHA2MLA Fine-Tuning

First, prepare a configuration file that can refer to the [MLA-FT configuration](../../configs/mla/rope_v4_topk4_svd_method7_rank16.yaml). Then, use the following command for MLA fine-tuning:

```bash
torchrun --nproc_per_node 4 \
    ../src/mha2mla_nt/run_train.py \
    --config-file ../configs/mla/rope_v4_topk4_svd_method7_rank16.yaml
```

| Partial-RoPE version | Strategy |
| :----: | --- |
| 0    | full-RoPE  |
| 1    | $\mathcal{S}_{\text{high}}$ |
| 2    | $\mathcal{S}_{\text{uniform}}$ |
| 4    | $\mathcal{S}_{\text{2-norm}}$ |
| 5    | $\mathcal{S}_{\text{low}}$ |

| SVD version | Strategy |
| :----: | --- |
| 2 |  $SVD_{split}$ |
| 7 |  $SVD_{joint}$ |

> If you want to use the partial-RoPE version 4, you should get the `qk_tensor` first.
> Using the following command, you can get the `qk_tensor`:
> ```bash
>torchrun --nproc_per_node 1 \
>    ../src/mha2mla_nt/2_norm.py \
>    --config-file ../configs/test/135M_2norm.yaml \
>    --output-dir ../utils/ \
>    --sample-size 1024
> ```

## Partial-RoPE Fine-Tuning For Ablation Experiment

If you want to conduct an ablation experiment using only partial-RoPE without SVD, you can use the following command. You just need to set the `SVD.version` to `0`. And use the same command Fine-Tuning. 

## Evaluation

We evaluate the model using the `transformers` framework after converting the checkpoint to Hugging Face (HF) format.

## Citation
```
@misc{ji2025economicalinferenceenablingdeepseeks,
      title={Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs}, 
      author={Tao Ji and Bin Guo and Yuanbin Wu and Qipeng Guo and Lixing Shen and Zhan Chen and Xipeng Qiu and Qi Zhang and Tao Gui},
      year={2025},
      eprint={2502.14837},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14837}, 
}
```
