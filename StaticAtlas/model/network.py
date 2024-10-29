import torch
import torch.nn as nn
from torch import Tensor
from model.svfnet import VecInt, SpatialTransformer

class RegistrationModelSVF(nn.Module):
    def __init__(self, model: nn.Module, inshape: [int, int, int], int_steps: int = 7):
        super().__init__()
        self.vecint = VecInt(inshape=inshape, nsteps=int_steps)
        self.spatial_transformer = SpatialTransformer(size=inshape)
        self.model = model

    def forward(self, source: Tensor, target: Tensor) -> (Tensor, Tensor):
        x = torch.cat([source, target], dim=1) #.to(torch.float32)
        forward_velocity = self.model(x)
        forward_flow = self.vecint(forward_velocity)
        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)
        return forward_flow, backward_flow


    def warped_image(self, tensor: Tensor, flow: Tensor) -> Tensor:
        return self.spatial_transformer(tensor, flow)

    @staticmethod
    def regularizer(disp, penalty='l2'):
        dy = torch.abs(disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        dx = torch.abs(disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        dz = torch.abs(disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0
