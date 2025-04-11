# Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    cache_file = f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    random.seed(seed)
    
    if not os.path.exists("cache"):
        os.makedirs("cache")
        
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        print(f"[Calib data] Load from {cache_file}")
        return traindataset
        
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
        
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        
    torch.save(traindataset, cache_file)
    return traindataset


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def get_whiten_scale_matrix(model, tokenizer, dev="cuda", calib_dataset="wikitext2", output_dir="cache/whiten"):
    model_id = model.config._name_or_path
    
    # Load calibration data
    calib_loader = get_calib_data(
        calib_dataset, 
        tokenizer, 
        model_id, 
        nsamples=256, 
        seqlen=2048
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    cache_file = f"{output_dir}/{model_id.replace('/','_')}_w2_scaling_matrices_fp16.pt"
    
    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """
    
    print(f"[whiten] Calibration dataset: {calib_dataset}")
    print(f"[whiten] Output file: {cache_file}")
    
    print(f"Creating whiten scale matrix dict...")

    # Create Scaling Matrix with low-resource inference
    # Adapted from https://github.com/AIoT-MLSys-Lab/SVD-LLM/blob/main/SVDLLM.py
    # Here, inference are performed in a layer-wise manner
    use_cache_orig = model.config.use_cache
    model.config.use_cache = False
    
    # Identify model architecture
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat" in model_id:
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model architecture: {model_id}")
        
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    # Setup caching mechanism
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    # Capture inputs
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    # Restore layer and free memory
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    scaling_matrices = []
    
    print("[Decomposition] Start to calculate the scaling matrix in layer-wise manner...")
    
    # Process each layer
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
            
        # Register hooks
        handles = []
        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
            
        # Process inputs
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), 
                           attention_mask=attention_masks, 
                           position_ids=position_ids[0].unsqueeze(0))[0]
                           
        # Remove hooks
        for h in handles:
            h.remove()
            
        # Move layer back to CPU
        layer = layer.cpu()
        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        
        # Process scaling matrices
        layer_scaling_matrices = {}
        for name in subset:
            if not ("k_proj" in name or "v_proj" in name):
                continue
                
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                subset[name].scaling_diag_matrix = scaling_diag_matrix
            except Exception:
                print("Eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    print("Raw scaling_diag_matrix is not a symmetric matrix!")
                    
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                
                if torch.isnan(scaling_diag_matrix).any():
                    print("Scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    print("Scaling_diag_matrix contains Inf!")
                    
                del eigenvalues
                subset[name].scaling_diag_matrix = scaling_diag_matrix
                
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception:
                print("Scaling_diag_matrix is not full rank!")
                reg_inv = 1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            
            del scaling_matrix_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.cpu()
            torch.cuda.empty_cache()
            
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
        
    # Restore model state
    model.config.use_cache = use_cache_orig
    
    # Save the scaling matrices
    torch.save(scaling_matrices, cache_file)
    print(f"Saved the whiten scale matrix dict to: {cache_file}")
        
    return scaling_matrices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate whitening scale matrices for LLM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Model ID or path to load")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2", 
                        choices=["wikitext2", "c4"], help="Calibration dataset to use")
    parser.add_argument("--output_dir", type=str, default="cache/whiten", 
                        help="Directory to save the scaling matrices")
    args = parser.parse_args()
    

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    
    print(f"Generating whitening scale matrices...")
    get_whiten_scale_matrix(
        model, 
        tokenizer, 
        calib_dataset=args.calib_dataset, 
        output_dir=args.output_dir
    )