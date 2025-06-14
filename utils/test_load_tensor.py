import torch


qk_rank_path = "qwen1.5_0.5B-2_norm_rank.pth"
with open(qk_rank_path, "rb") as fin:
    qk_norm_rank = torch.load(fin, weights_only=True)
    # [n_layers, n_key_value_heads, d_head]
    print(qk_norm_rank.size())  # torch.Size([24, 16, 64])

    # same RoPE freqs have same rank
    n_freqs = qk_norm_rank.size(-1) // 2
    freqs_group_1 = qk_norm_rank[..., :n_freqs]
    freqs_group_2 = qk_norm_rank[..., n_freqs:]
    print(torch.eq(freqs_group_1, freqs_group_2))  # True

    # rank from 0 to n_freqs-1
    layer_id = 0
    head_id = 0
    print(qk_norm_rank[layer_id][head_id])
    # [31, 30, 27, 26, 24, 22, 16, 17, 12, 14, 18, 9, 20, 29, 21, 15, 23, 19, 10, 5, 13, 7, 11, 25, 28, 3, 1, 4, 6, 2, 0, 8,
    #  31, 30, 27, 26, 24, 22, 16, 17, 12, 14, 18, 9, 20, 29, 21, 15, 23, 19, 10, 5, 13, 7, 11, 25, 28, 3, 1, 4, 6, 2, 0, 8]
