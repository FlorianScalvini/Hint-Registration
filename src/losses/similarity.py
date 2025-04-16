import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NCCLoss(nn.Module):
    """
    Computes the global Normalized Cross-Correlation (NCC) coefficient between a batch of input images and
    a batch of output images

    Args:
        input_image (torch.Tensor): [B, 1, D, H, W]
        target_image (torch.Tensor): [B, 1, D, H, W]

    Returns:
        torch.Tensor: the NCC between the input and target images
    """
    def __init__(self) -> None:
        super().__init__()



    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        input_batch_mean = torch.mean(pred.flatten(start_dim=1), dim=-1).view(-1, 1, 1, 1)
        target_batch_mean = torch.mean(gt.flatten(start_dim=1), dim=-1).view(-1, 1, 1, 1)

        numerator = (pred - input_batch_mean) * (gt - target_batch_mean)
        numerator = torch.sum(numerator.flatten(start_dim=1), dim=-1)

        denum1 = torch.sum(((pred - input_batch_mean) ** 2).flatten(start_dim=1), dim=-1)
        denum2 = torch.sum(((gt - target_batch_mean) ** 2).flatten(start_dim=1), dim=-1)
        denum = torch.sqrt(denum1 * denum2)

        return torch.mean(numerator / denum)


