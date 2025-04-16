# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/sgl-project/sglang/blob/f8194b267c71d79fca30e0bd8d01c086969abd4e/python/sglang/srt/layers/attention/triton_ops/decode_attention.py



import logging

import triton
import triton.language as tl
from flash_attn import flash_attn_with_kvcache
import torch
from functools import partial
logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
logger.warning(
    "The following error message 'operation scheduled before its operands' can be ignored."
)


_MIN_BLOCK_KV = 32


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q_NOPE,
    Q_PE,
    KV_NOPE_Buffer,
    K_PE_Buffer,
    sm_scale,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_q_nope_bs,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_buf_kv_nope_bs,
    stride_buf_kv_nope_ls,
    stride_buf_kv_nope_h,
    stride_buf_kv_pe_bs,
    stride_buf_kv_pe_ls,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_seq_len: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    d_pe: tl.constexpr,
    d_nope: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d_nope = tl.arange(0, BLOCK_DNOPE)
    # Assuming that on the PE part, query of different heads are flattened
    offs_d_pe = tl.arange(0, BLOCK_DPE*q_head_num)
    #mask_d_nope = offs_d_nope < d_nope
    #mask_d_pe = offs_d_pe < d_pe

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q_nope = cur_batch * stride_q_nope_bs + cur_head[:, None] * stride_q_nope_h + offs_d_nope[None, :]
    #offs_q_pe = cur_batch * stride_q_pe_bs + cur_head[:, None] * stride_q_pe_h + offs_d_pe[None, :]
    offs_q_pe = cur_batch * stride_q_pe_bs + offs_d_pe[None, :]

    
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    partial_qk_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    partial_exp_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DNOPE], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q_nope = tl.load(Q_NOPE + offs_q_nope, mask=(mask_h[:, None]), other=0.0)
        q_pe = tl.load(
            Q_PE + offs_q_pe #[1, BLOCK_DPE*q_head_num]
        )
        #tl.static_print("q_pe", q_pe)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # Load K to K^T
            offs_buf_kv_nope = (
                cur_batch * stride_buf_kv_nope_bs
                + offs_n[None, :] * stride_buf_kv_nope_ls 
                + cur_kv_head * stride_buf_kv_nope_h
                + offs_d_nope[:, None]
            )
            kv_nope = tl.load(
                KV_NOPE_Buffer + offs_buf_kv_nope,
                mask=(offs_n[None, :] < split_kv_end),
                other=0.0,
            )
            qk = tl.dot(q_nope, kv_nope.to(q_nope.dtype))
            
            # RoPE Parts (Not finished yet)
            offs_buf_kpe = (
                cur_batch * stride_buf_kv_pe_bs
                + offs_n[:, None] * stride_buf_kv_pe_ls
                + offs_d_pe[None, :]
            )
            k_pe = tl.load(
                K_PE_Buffer + offs_buf_kpe,
                mask=(offs_n[:, None] < split_kv_end),
                other=0.0,
            ) #[BLOCK_N, BLOCK_DPE*q_head_num]
            q_pe_k_pe = q_pe * k_pe #[BLOCK_N, BLOCK_DPE*q_head_num]
            q_pe_k_pe = tl.reshape(q_pe_k_pe, [BLOCK_N, q_head_num, BLOCK_DPE]) #[BLOCK_N, q_head_num, BLOCK_DPE]
            q_pe_k_pe = tl.sum(q_pe_k_pe, 2) #[BLOCK_N, q_head_num]
            qk += tl.trans(q_pe_k_pe)

            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            new_partial_qk_max = tl.maximum(tl.max(qk, 1), partial_qk_max)
            re_scale = tl.exp(partial_qk_max - new_partial_qk_max)
            p = tl.exp(qk - new_partial_qk_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(kv_nope.dtype), tl.trans(kv_nope))

            partial_exp_sum = partial_exp_sum * re_scale + tl.sum(p, 1)
            partial_qk_max = new_partial_qk_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d_nope[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / partial_exp_sum[:, None],
            mask=(mask_h[:, None]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // d_nope

        tl.store(
            Att_Lse + offs_mid_o_1,
            partial_qk_max + tl.log(partial_exp_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q_nope, # shape: [batch, num_q_heads, head_dim]
    q_pe, # shape: [batch, num_q_heads*pe_dim]
    kv_nope_buffer, # shape: [batch, kv_len, num_kv_heads, head_dim]
    kv_pe_buffer, # shape: [batch, kv_len, num_q_heads*pe_dim]
    attn_out,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    BLOCK_DNOPE = kv_nope_buffer.shape[-1]
    BLOCK_DPE = kv_pe_buffer.shape[-1] // q_nope.shape[1]
    d_nope = kv_nope_buffer.shape[-1]
    d_pe = kv_pe_buffer.shape[-1]
    batch, head_num = q_nope.shape[0], q_nope.shape[1]
    kv_group_num = q_nope.shape[1] // kv_nope_buffer.shape[2]
    kv_seq_len = kv_nope_buffer.shape[1]
    BLOCK_H = 32 # NUMBER OF HEADS for each thread block
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    #print(d_nope, d_pe, head_num, kv_group_num, kv_seq_len)
    _fwd_grouped_kernel_stage1[grid](
        Q_NOPE=q_nope,
        Q_PE=q_pe,
        KV_NOPE_Buffer=kv_nope_buffer,
        K_PE_Buffer=kv_pe_buffer,
        sm_scale=sm_scale,
        Att_Out=attn_out,
        Att_Lse=attn_lse,
        num_kv_splits=num_kv_splits,
        stride_q_nope_bs=q_nope.stride(0),
        stride_q_nope_h=q_nope.stride(1),
        stride_q_pe_bs=q_pe.stride(0),
        stride_buf_kv_nope_bs=kv_nope_buffer.stride(0),
        stride_buf_kv_nope_ls=kv_nope_buffer.stride(1),
        stride_buf_kv_nope_h=kv_nope_buffer.stride(2),
        stride_buf_kv_pe_bs=kv_pe_buffer.stride(0),
        stride_buf_kv_pe_ls=kv_pe_buffer.stride(1),
        stride_mid_ob=attn_out.stride(0),
        stride_mid_oh=attn_out.stride(1),
        stride_mid_os=attn_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        kv_seq_len=kv_seq_len,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DNOPE=BLOCK_DNOPE,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        d_pe=d_pe,
        d_nope=d_nope,
        num_warps=4,
        num_stages=num_stages,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    attn_logits,
    attn_lse,
    o,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    kv_seq_len: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_seq_len = kv_seq_len
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    global_exp_sum = 0.0
    global_lse = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_kv_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                attn_logits + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            in_coming_lse = tl.load(attn_lse + offs_logic + split_kv_id * stride_mid_os // Lv)
            new_global_lse = tl.maximum(in_coming_lse, global_lse)
            old_scale = tl.exp(global_lse - new_global_lse)
            acc *= old_scale
            exp_logic = tl.exp(in_coming_lse - new_global_lse)
            acc += exp_logic * tv

            global_exp_sum = global_exp_sum * old_scale + exp_logic
            global_lse = new_global_lse

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / global_exp_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    attn_logits,
    attn_lse,
    q,
    o,
    v_buffer,
    num_kv_splits,
    max_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    kv_seq_len = v_buffer.shape[1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits

    extra_kargs = {}
    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        attn_logits,
        attn_lse,
        o,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        kv_seq_len=kv_seq_len,
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )

def decode_attention_fwd(
    q_nope,             # shape: [batch, num_q_heads, head_dim]
    q_pe,               # shape: [batch, num_q_heads, pe_dim]
    kv_nope_buffer,     # shape: [batch, kv_len, num_kv_heads, head_dim]
    kv_pe_buffer,       # shape: [batch, kv_len, num_q_heads, pe_dim]
    sm_scale=1.0,       # float: softmax scaling factor
    logit_cap=0.0,      # float: cap for logits
):
    #print(q_pe.shape)
    #print(q_nope.shape)
    #print(kv_nope_buffer.shape)
    #print(kv_pe_buffer.shape)
    batch, num_q_heads, head_dim = q_nope.shape
    _, kv_len, num_kv_heads, _ = kv_nope_buffer.shape
    
    # Calculate kv_group_num (ratio of query heads to key/value heads)
    kv_group_num = num_q_heads // num_kv_heads
    assert kv_group_num >= 1
    
    # Determine number of KV splits if not provided
    # FIXME(brian1009): Determine the optimal number of KV splits on the fly
    num_kv_splits = torch.ones(batch, dtype=torch.int32, device=q.device) * 64
    max_kv_splits = 64

    # Create intermediate tensors for attention computation
    attn_logits = torch.empty(
        (batch, num_q_heads, max_kv_splits, head_dim),
        dtype=torch.float32, 
        device=q.device
    )
    
    attn_lse = torch.empty(
        (batch, num_q_heads, max_kv_splits),
        dtype=torch.float32,
        device=q.device
    )

    _decode_grouped_att_m_fwd(
        q_nope=q_nope,
        q_pe=q_pe,
        kv_nope_buffer=kv_nope_buffer,
        kv_pe_buffer=kv_pe_buffer,
        attn_out=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )
    o = torch.empty(
        (batch, num_q_heads, head_dim),
        dtype=torch.float32,
        device=q.device
    )

    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q_nope,
        o,
        v_buffer=kv_nope_buffer,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
    )

    return o


import torch
import torch.nn.functional as F
import argparse
import argparse


def mha2mla_ref(q, q_pe, kv, k_pe):
    """
    Inputs:
    - q (Tensor): [batch, q_head_num, q_len=1, dim]
    - q_pe (Tensor): [batch, q_head_num, q_len=1, pe_dim]
    - kv (Tensor): [batch, kv_head_num=1, kv_len, dim]
    - k_pe (Tensor): [batch, q_head_num, kv_len, pe_dim]  
    Outputs:
    - output (Tensor): [batch, q_head_num, dim]
    """
    batch = q.shape[0]
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    #scale = (dim + pe_dim)**0.5
    scale = 1.0
    seqlen_q = q.shape[1]
    kv_head_num = kv.shape[1]
    q_head_num = q.shape[2]
    assert seqlen_q == 1, "assuming the decoding stage. seqlen_q should be 1"
    assert kv_head_num == 1, "assuming kv_head_num is 1"
    num_head_groups = q_head_num // kv_head_num

    # part 1: scores  
    ## A. NoPE parts (GEMM)
    q = q.view(batch, kv_head_num, num_head_groups*1, dim) # [batch, kv_head_num, num_head_groups*1, dim]
    kv_t = kv.transpose(-1, -2) # [batch, kv_head_num, dim, kv_len]
    scores_no_rope = torch.matmul(q, kv_t) # [batch, kv_head_num, num_head_groups*1, kv_len] Ex: (bsz, 1, 32, 8192)
    
    ## B. RoPE parts (Batch GEMV)
    q_pe = q_pe.view(batch, num_head_groups*1, kv_head_num, pe_dim) # [batch, kv_head_num, num_head_groups*1, pe_dim]
    kv_pe_t = k_pe.transpose(-1, -2) # [batch, q_head_num, pe_dim, kv_len]
    scores_rope = torch.matmul(q_pe, kv_pe_t) # [batch, q_head_num, q_len, kv_len] Ex: (bsz, 32, 1, 8192)
    
    ## C. Combine scores
    scores = scores_no_rope.transpose(1, 2) + scores_rope # [batch, q_head_num, q_len, kv_len] Ex: (bsz, 32, 1, 8192)
    scores = scores / scale # [batch, q_head_num, q_len, kv_len] Ex: (bsz, 32, 1, 8192)

    # part 2: Apply softmax
    attention = F.softmax(scores, dim=-1)  # [batch, q_head_num, q_len, kv_len] Ex: (bsz, 32, 1, 8192)
    # part 3: Compute output
    out = torch.matmul(attention, kv) # [batch, q_head_num, q_len, dim] Ex: (bsz, 32, 1, 512)
    out = out.squeeze(2) # [batch, q_head_num, dim] Ex: (bsz, 32, 512)
    return out

def flash_attn_mha_fwd(
        q, # shape: [batch, q_len, q_head_num, dim]
        k, # shape: [batch, kv_len, kv_head_num, dim]
        v, # shape: [batch, kv_len, kv_head_num, dim]   
    ):
    return flash_attn_with_kvcache(
        q,
        k,
        v,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='q heads number')
    parser.add_argument('--kv_heads', type=int, default=1, help='kv heads number')
    parser.add_argument('--kv_ctx', type=int, default=16384, help='kv context length')
    parser.add_argument('--dim', type=int, default=512, help='head dim')
    parser.add_argument('--pe_dim', type=int, default=16, help='pe head dim')
    args = parser.parse_args()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim
    
    # Create random tensors for inputs
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)  # [batch, q_len, q_head_num, dim]
    q_pe = torch.randn(batch, heads, pe_dim, device="cuda", dtype=torch.float16)  # [batch, q_len, q_head_num, pe_dim]
    kv = torch.randn(batch, kv_heads, kv_ctx, dim, device="cuda", dtype=torch.float16)  # [batch, kv_head_num, kv_len, dim]
    k_pe = torch.randn(batch, heads, kv_ctx, pe_dim, device="cuda", dtype=torch.float16)  # [batch, q_head_num, kv_len, pe_dim]
    
    # Run the reference program
    ref_output = mha2mla_ref(q.unsqueeze(1), q_pe.unsqueeze(1), kv, k_pe)

    out = decode_attention_fwd(
        q_nope=q.squeeze(1), 
        q_pe=q_pe.reshape(batch, heads*pe_dim), 
        kv_nope_buffer=kv.transpose(1, 2).contiguous(), 
        kv_pe_buffer=k_pe.transpose(1, 2).reshape(batch, kv_ctx, heads*pe_dim).contiguous()
    )
    #print(out)
    print("Output from Reference: ", ref_output)
    print("Output from Triton: ", out)

    print("Difference: ", torch.max(torch.abs(ref_output - out)))


    # Add timing code to compare performance between reference and Triton implementations
    import time
    
    def measure_time(func, *args, warmup=10, repeat=100):
        """
        Measure the execution time of a function.
        
        Args:
            func: Function to measure
            args: Arguments to pass to the function
            warmup: Number of warmup iterations
            repeat: Number of measurement iterations
        
        Returns:
            Average execution time in milliseconds
        """
        # Warmup
        for _ in range(warmup):
            func(*args)
        
        torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        for _ in range(repeat):
            func(*args)
        torch.cuda.synchronize()
        end = time.time()
        
        return (end - start) * 1000 / repeat  # Convert to ms
    
    print("\n=== Performance Measurements ===")
    
    q_ = q.unsqueeze(1)
    q_pe_ = q_pe.unsqueeze(1)
       
    ## Reference MHA2MLA (Torch)
    time_ref = measure_time(
        partial(mha2mla_ref,
            q=q_, 
            q_pe=q_pe_, 
            kv=kv, 
            k_pe=k_pe
        )
    )
    
    ## Triton MHA2MLA
    q__ = q.squeeze(1)
    q_pe__ = q_pe.reshape(batch, heads*pe_dim)
    kv_nope_buffer__ = kv.transpose(1, 2).contiguous()
    kv_pe_buffer__ = k_pe.transpose(1, 2).reshape(batch, kv_ctx, heads*pe_dim).contiguous()
    # Measure Triton implementation
    time_triton = measure_time(
        partial(decode_attention_fwd,
            q_nope=q__, 
            q_pe=q_pe__, 
            kv_nope_buffer=kv_nope_buffer__, 
            kv_pe_buffer=kv_pe_buffer__
        )
    )
    
    ## Flash2 MHA
    q_fa = torch.randn(batch, 1, heads, 128, device="cuda", dtype=torch.float16)
    k_fa = torch.randn(batch, kv_ctx, 32, 128, device="cuda", dtype=torch.float16)
    v_fa = torch.randn(batch, kv_ctx, 32, 128, device="cuda", dtype=torch.float16)
    time_flash = measure_time(
        partial(flash_attn_mha_fwd,
            q=q_fa, k=k_fa, v=v_fa
        )
    )
    print(f"Reference implementation: {time_ref:.4f} ms")
    print(f"Triton implementation: {time_triton:.4f} ms")
    print(f"Flash implementation: {time_flash:.4f} ms")
    print(f"Speedup (Triton MHA2MLA vs Torch MHA2MLA): {time_ref / time_triton:.2f}x")
    print(f"Speedup (Flash2 MHA vs Triton MHA2MLA): {time_flash / time_triton:.2f}x")
