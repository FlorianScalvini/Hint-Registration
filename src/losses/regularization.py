from __future__ import absolute_import
import torch
import torch.nn as nn

class Grad3d(nn.Module):
    def __init__(self, penalty='l1'):
        super().__init__()
        if penalty not in ['l1', 'l2']:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.penalty = penalty

    def forward(self, x_pred):
        # Compute gradients in each direction
        dx = torch.abs(x_pred[:, :, 1:, :, :] - x_pred[:, :, :-1, :, :])
        dy = torch.abs(x_pred[:, :, :, 1:, :] - x_pred[:, :, :, :-1, :])
        dz = torch.abs(x_pred[:, :, :, :, 1:] - x_pred[:, :, :, :, :-1])
        # Apply penalty (squared for L2 penalty)
        if self.penalty == 'l2':
            dy, dx, dz = dy**2, dx**2, dz**2
        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        return grad

class MagnitudeLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super().__init__()
        loss_fn = {'l1': nn.L1Loss, 'l2': nn.MSELoss}
        if penalty not in loss_fn:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.loss = loss_fn[penalty]()

    def forward(self, x):
        return self.loss(x, torch.zeros_like(x))

