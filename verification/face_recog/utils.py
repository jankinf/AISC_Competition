import torch

def l2_dis(feat1: torch.Tensor, feat2: torch.Tensor):
    return torch.norm(feat1 - feat2, p=2)

def l2_sigmoid(feat1: torch.Tensor, feat2: torch.Tensor, alpha: float=20, beta: float=-3.2):
    dis_l2 = l2_dis(feat1, feat2)
    dis_l2_sig = 1 / (1 + torch.exp(alpha * (dis_l2 ** 2) + beta)) * 100
    return dis_l2_sig