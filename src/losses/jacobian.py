import torch
import torch.nn as nn
from typing import Union, Sequence
import pystrum.pynd.ndutils as nd
import numpy as np
import torch.nn.functional as F

def Get_Ja(displacement):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    displacement = displacement.squeeze().permute(1, 2, 3, 0)
    # check inputs
    volshape = displacement.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    # compute gradients
    J = torch.gradient(displacement + torch.Tensor(grid).to(displacement.device))

    dx = J[0]
    dy = J[1]
    dz = J[2]

    # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    return Jdet0 - Jdet1 + Jdet2

def compute_jacobian_determinant(J):
    # Assume J is in normalized coordinates [-1, 1]
    # Step 1: Normalize from [-1, 1] → [0, 1]
    J = J + 1          #恢复到0到1
    J = J / 2.
    scale_factor = torch.tensor([J.size(1), J.size(2), J.size(3)]).to(J).view(1, 1, 1, 1, 3) * 1.
    J = J * scale_factor

    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def jacobian_determinant_3d(deformed_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of the Jacobian numerically, given the deformed
    output grid and returns the percentage of negative values

    Args:
        deformed_grid (torch.Tensor): [B, D, H, W, 3]

    Returns:
        torch.Tensor: the percentage of negative determinants
    """
    dy = deformed_grid[:, 1:, :-1, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dx = deformed_grid[:, :-1, 1:, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dz = deformed_grid[:, :-1, :-1, 1:, :] - deformed_grid[:, :-1, :-1, :-1, :]

    det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    determinants = det0 - det1 + det2

    return determinants


class Jacobianloss(nn.Module):
    """
    Jacobian loss for penalizing the Jacobian determinant of a displacement field.
    This loss can be used to ensure that the transformation is invertible and smooth.
    """
    def __init__(self):
        super(Jacobianloss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Penalizing Jacobian
        Args:
            x (torch.Tensor): Displacement field of shape (B, C, D, H, W) or (B, C, H, W).
            spacing (Union[list[int], tuple[int]]): Spacing of the input image.
        Returns:
            torch.Tensor: Jacobian loss value.
        '''

        Jdet = compute_jacobian_determinant(x)
        Neg_Jac = 0.5 * (torch.abs(Jdet) - Jdet)
        return torch.sum(Neg_Jac)
