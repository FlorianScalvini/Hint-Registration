
import torch
import torch.nn as nn
from monai.networks.blocks.warp import Warp


class InverseConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.warp = Warp()

    def forward(self, phi_a, phi_b):
        identity = self.identity_map(phi_a.shape)
        appro_for = self.warp(phi_a, phi_b) - identity
        appro_inv = self.warp(phi_b, phi_a) - identity
        return  appro_for**2 + appro_inv**2

    @staticmethod
    def identity_map(size):
        dim = len(size[2:])
        if dim != size[1]:
            raise ValueError(
                f"Number of spatial dimensions (sz[2:]) should match the "
                f"second dimension (channels), got {dim} != {size[1]}")
        spacing = 1.0 / (torch.tensor(size[2:]) - 1)
        grids = [torch.arange(0, size[i + 2]) for i in range(dim)]
        mesh = torch.meshgrid(*grids)  # Use 'ij' indexing for matrix-style indexing
        coordinates = torch.stack(mesh, dim=0).float()
        for d in range(dim):
            coordinates[d] *= spacing[d]
        id_map = torch.zeros(size, dtype=torch.float)
        for b in range(size[0]):
            for c in range(dim):
                id_map[b, c, ...] = coordinates[c]
        return id_map

class IconInverseConsistency(InverseConsistency):
    def __init__(self):
        super().__init__()

    def forward(self, phi_a, phi_b):
        identity = self.identity_map(phi_a.shape)
        eps_identity = (identity + torch.randn(*identity.shape) / identity.shape[-1]).to(phi_a.device)
        approx_ab = self.warp(self.warp(eps_identity, phi_a), phi_b)
        approx_ba = self.warp(self.warp(eps_identity, phi_b), phi_a)
        loss = (torch.mean((eps_identity - approx_ab) ** 2)  + torch.mean((eps_identity - approx_ba) ** 2))
        return loss


class GradIconInverseConsistency(InverseConsistency):
    def __init__(self):
        super().__init__()
        self.delta = 0.001

    def forward(self, phi_a, phi_b):
        identity = self.identity_map(phi_a.shape)
        eps_identity = (identity + torch.randn(*identity.shape) / identity.shape[-1]).to(phi_a.device)
        approx_ba = self.warp(self.warp(eps_identity, phi_b), phi_a)
        inverse_consistency_error = eps_identity - approx_ba

        if len(self.identity.shape) == 4:
            dx = torch.Tensor([[[[self.delta]], [[0.0]]]]).to(phi_a.device)
            dy = torch.Tensor([[[[0.0]], [[self.delta]]]]).to(phi_a.device)
            direction_vectors = (dx, dy)
        else:
            dx = torch.Tensor([[[[[self.delta]]], [[[0.0]]], [[[0.0]]]]]).to(phi_a.device)
            dy = torch.Tensor([[[[[0.0]]], [[[self.delta]]], [[[0.0]]]]]).to(phi_a.device)
            dz = torch.Tensor([[[[[0.0]]], [[[0.0]]], [[[self.delta]]]]]).to(phi_a.device)
            direction_vectors = (dx, dy, dz)

        direction_losses = []
        for d in direction_vectors:
            approx_ba_d = self.warp(self.warp(eps_identity + d, phi_b), phi_a)
            inverse_consistency_error_d = eps_identity + d - approx_ba_d
            grad_d_icon_error = (inverse_consistency_error - inverse_consistency_error_d) / self.delta
            direction_losses.append(torch.mean(grad_d_icon_error ** 2))
        loss = sum(direction_losses)
        return loss
