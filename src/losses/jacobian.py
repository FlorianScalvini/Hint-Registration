import torch
import torch.nn as nn
from typing import Union, Sequence

def calculate_jacobian(u: torch.Tensor, spacing: Union[Sequence[int], tuple[Sequence[int]]]) -> torch.Tensor:
    """
    Calculate the Jacobian of a displacement field.
    Args:
        u (torch.Tensor): Displacement field of shape (B, C, D, H, W) or (B, C, H, W).
        spacing (Union[list[int], tuple[int]]): Spacing of the input image.
    Returns:
        torch.Tensor: Jacobian of the displacement field.
    """
    B, C, D, H, W = u.shape
    if C != 3 and C != 2:
        raise ValueError("Displacement field must be 2D or 3D")
    jacobians = []
    for b in range(B):
        grads = [torch.gradient(u[b, i], spacing=spacing) for i in range(C)]
        J = torch.stack(
            [torch.stack([grads[i][j] + (i == j) for j in range(C)], dim=-1) for i in range(C)],
            dim=-1
        )
        jacobians.append(J)
    return torch.stack(jacobians)

def determinant_jacobian(u: torch.Tensor, spacing: Union[list[int], tuple[int]]) -> torch.Tensor:
    """
    Compute the determinant of the Jacobian matrix of a displacement field.

    Args:
        u (torch.Tensor): Displacement field of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D.
        spacing (Union[list[float], tuple[float]]): Voxel spacing in each spatial dimension.

    Returns:
        torch.Tensor: Determinant of the Jacobian matrix with shape (B, D, H, W) for 3D
                      or (B, H, W) for 2D.
    """
    return torch.linalg.det(calculate_jacobian(u, spacing))


def jacobian_determinant_3d(deformed_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of the Jacobian numerically, given the deformed
    output grid and returns the percentage of negative values

    Args:
        deformed_grid (torch.Tensor): [B, D, H, W, 3]

    Returns:
        torch.Tensor: the percentage of negative determinants
    """
    dy = deformed_grid[:, :, 1:, :-1, :-1,] - deformed_grid[:, :, -1, :-1, :-1]
    dx = deformed_grid[:, :, -1, 1:, :-1] - deformed_grid[:, :, -1, :-1, :-1]
    dz = deformed_grid[:, :, -1, :-1, 1:] - deformed_grid[:, :, -1, :-1, :-1]

    det0 = dx[:, 0, :, :, :] * (dy[:, 1, :, :, :] * dz[:, 2, :, :, :] - dy[:, 2, :, :, :] * dz[:, 1, :, :, :])
    det1 = dx[:, 1, :, :, :] * (dy[:, 2, :, :, :] * dz[:, 2, :, :, :] - dy[:, 2, :, :, :] * dz[:, 0, :, :, :])
    det2 = dx[:, 2, :, :, :] * (dy[:, 0, :, :, :] * dz[:, 1, :, :, :] - dy[:, 1, :, :, :] * dz[:, 0, :, :, :])

    determinants = det0 - det1 + det2

    num_neg_dets = len(determinants[determinants <= 0])
    total_points = torch.prod(torch.tensor(determinants.size(), device=determinants.device))

    neg_dets_percentage = num_neg_dets * 100 / total_points

    return neg_dets_percentage


class Jacobianloss(nn.Module):
    """
    Jacobian loss for penalizing the Jacobian determinant of a displacement field.
    This loss can be used to ensure that the transformation is invertible and smooth.
    """
    def __init__(self):
        super(Jacobianloss, self).__init__()

    def forward(self, x: torch.Tensor, spacing: Union[list[int], tuple[int]] = (1, 1, 1)) -> torch.Tensor:
        '''
        Penalizing Jacobian
        Args:
            x (torch.Tensor): Displacement field of shape (B, C, D, H, W) or (B, C, H, W).
            spacing (Union[list[int], tuple[int]]): Spacing of the input image.
        Returns:
            torch.Tensor: Jacobian loss value.
        '''
        return (torch.max(-determinant_jacobian(x, spacing), torch.zeros_like(x))**2).sum()

