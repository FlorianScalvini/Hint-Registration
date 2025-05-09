import monai.networks.blocks
import torch
import torch.nn as nn
from torch import Tensor
from monai import networks
import torch.nn.functional as F
from typing import Union
import numpy as np

class RegistrationModule(nn.Module):
    '''
        Registration module for 3D image registration.yaml with DVF
    '''
    def __init__(self):
        '''
        :param model: nn.Module
        '''
        super().__init__()

    def forward(self, data: Tensor) -> Tensor:
        '''
            Forward pass of the registration module
            :param data: Input images
            :return: Deformation field
        '''
        return self.model(data)


    def load_network(self, path) -> None:
        '''
            Load the network weights
            :param path: Path to the weights
        '''
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def warp(image: Tensor, flow: Tensor) -> Tensor:
        """
        Warp an image using a dense flow field.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) or (B, C, D, H, W).
            flow (Tensor): Flow field of the same shape as image (excluding channel), shape (B, D, H, W, 3) or (B, H, W, 2).
            mode (str): Interpolation mode ('bilinear' or 'nearest').

        Returns:
            Tensor: Warped image.
        """
        warp = monai.networks.blocks.Warp(mode='bilinear', padding_mode='zeros')
        return warp(image, flow)


    @staticmethod
    def compose(flow_a: torch.Tensor, flow_b: torch.Tensor) -> torch.Tensor:
        warp = monai.networks.blocks.Warp(mode='bilinear', padding_mode='zeros')
        return warp(flow_b, flow_a) + flow_a



    @staticmethod
    def flow2phi(flow: Tensor, grid_normalize=False) -> torch.Tensor:
        """
        Convert a flow field to a normalized sampling grid (phi) for grid_sample.

        Args:
            flow (Tensor): Flow field of shape (B, D, H, W, 3) or (B, H, W, 2).

        Returns:
            Tensor: Normalized grid (phi) suitable for F.grid_sample.
            :param grid_normalize:
        """
        grid = RegistrationModule._make_identity_grid(flow.shape).to(flow.device)
        phi = flow + grid
        shape = flow.shape[2:]
        if grid_normalize:
            for i in range(len(shape)):
                phi[:, i, ...] = 2 * (phi[:, i, ...] / (shape[i] - 1) - 0.5)
        # move channels dim to last position
        if len(shape) == 2:
            phi = phi.permute(0, 2, 3, 1)
            phi = phi[..., [1, 0]]
        elif len(shape) == 3:
            phi = phi.permute(0, 2, 3, 4, 1)
            phi = phi[..., [2, 1, 0]]
        return phi

    @staticmethod
    def _make_identity_grid(shape: torch.Tensor.shape) -> torch.Tensor:
        spatial_dims = shape[2:]
        coords = [torch.arange(0, s) for s in spatial_dims]
        mesh = torch.meshgrid(*coords, indexing='ij')  # Ensures correct axis order
        grid = torch.stack(mesh, dim=0).float()  # (C, ...)
        grid = grid.unsqueeze(0)  # Add batch dimension: (1, C, ...)
        return grid

