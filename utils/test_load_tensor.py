import pickle
import torch


with open("qk_tensor_135M.pth", "rb") as fin:
    qk_tensor = torch.load(fin)  # torch.Size([30, 9, 32])
    qk_tensor = qk_tensor.view(30, 3, 3, 32).sum(dim=2)  # torch.Size([30, 3, 32])

    print(torch.any(qk_tensor < 0))
    topk_indices = torch.topk(input=qk_tensor[0], k=8, dim=1)[1]
    mask_matrix = torch.zeros_like(qk_tensor[0])
    mask_matrix.scatter_(1, topk_indices, 1)
    for x1, x2 in zip(qk_tensor[0][0], mask_matrix[0]):
        print(x1, x2)