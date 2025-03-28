from pathlib import Path
from transformers import AutoModelForCausalLM,LlamaForCausalLM
import torch

import os
import dataclasses
import nanotron
import argparse
from nanotron.config import LlamaConfig as NanotronLlamaConfig

from ..conversation.convert_weights import load_nanotron_model
from ..conversation.convert_nanotron_to_hf import get_hf_config,convert_nt_to_hf
from ..conversation.convert_hf_to_nanotron import convert_hf_to_nt
from .init import AttnForTraing

def merge(original_model_path:str,ae_model_path:str,save_path:str,save_in_nt_format:bool=False):
    from .patch_func_hf import ae_patch_func_hf
    from .patch_func_nt import ae_patch_func_nt,CustomLlamaConfig
    import json
    with open(os.path.join(ae_model_path,"config.json"),"r") as f:
        config = json.load(f)
    ae_patch_func_hf(config["RoPE"])
    ae_patch_func_nt(config["RoPE"])
    globals()["NanotronLlamaConfig"] = CustomLlamaConfig
    # Load from nt ckpt
    with open(os.path.join(original_model_path , "model_config.json"), "r") as f:
        attrs = json.load(f)
        model_config = NanotronLlamaConfig(**attrs)
    setattr(model_config,"RoPE",config["RoPE"])
    setattr(model_config,"AE",config["AE"])
    original_model_nt = load_nanotron_model(
        model_config=model_config,
        checkpoint_path=Path(original_model_path),
    )
    # Init hf model
    model_config_hf = get_hf_config(model_config)
    original_model_hf = LlamaForCausalLM._from_config(model_config_hf)
    convert_nt_to_hf(original_model_nt, original_model_hf, model_config)
    # Load ae model
    model = AttnForTraing.from_pretrained(ae_model_path)
    with torch.no_grad():
        for layer_idx, layer in enumerate(original_model_hf.model.layers):
            attn = model.model[layer_idx]
            layer.self_attn.auto_encoder.W_down_k.weight.data[:] = attn.auto_encoder.W_down_k.weight.detach()
            layer.self_attn.auto_encoder.W_up_k.weight.data[:] = attn.auto_encoder.W_up_k.weight.detach()
            assert hasattr(layer.self_attn.auto_encoder,"W_down_v") == hasattr(attn.auto_encoder,"W_down_v")
            if hasattr(layer.self_attn.auto_encoder,"W_down_v"):
                layer.self_attn.auto_encoder.W_down_v.weight.data[:] = attn.auto_encoder.W_down_v.weight.detach()
                layer.self_attn.auto_encoder.W_up_v.weight.data[:] = attn.auto_encoder.W_up_v.weight.detach()
    del model
    # save nt model
    if save_in_nt_format:
        # convert hf to nt
        convert_hf_to_nt(original_model_hf, original_model_nt, model_config)
        del original_model_hf
        parallel_context = nanotron.parallel.ParallelContext(
            data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
        )
        nanotron.serialize.save_weights(model=original_model_nt, parallel_context=parallel_context, root_folder=Path(save_path))
        with open(Path(save_path) / "model_config.json", "w+") as f:
            json.dump(dataclasses.asdict(model_config), f)
        print(f"NT format model saved to {save_path}")
    else:
        original_model_hf.save_pretrained(save_path)
        print(f"HF format model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model_path",type=str,required=True)
    parser.add_argument("--ae_model_path",type=str,required=True)
    parser.add_argument("--save_path",type=str,required=True)
    parser.add_argument("--save_in_nt_format",action="store_true")
    args = parser.parse_args()
    merge(args.original_model_path,args.ae_model_path,args.save_path,args.save_in_nt_format)

if __name__ =="__main__":
    main()
