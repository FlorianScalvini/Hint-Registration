import torch
import monai

class Composition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.warp = monai.networks.blocks.Warp(mode='bilinear', padding_mode='zeros')

    def forward(self, flow1: torch.Tensor, flow2: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Compose two flow fields using MONAI's warp.
        Args:
            flow1: [B, 3, D, H, W] - the second deformation (applied after flow2)
            flow2: [B, 3, D, H, W] - the first deformation
            grid: [B, 3, D, H, W] - identity grid (in voxel coordinates)

        Returns:
            Composed flow: [B, 3, D, H, W]
        """
        # Warp flow1 using MONAI's warp at displaced positions
        flow1_warped = monai.networks.blocks.Warp(mode='bilinear', padding_mode='zeros')(flow1, flow2)
        composed = flow1_warped + flow2
        return composed