import torch
import torchio as tio
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from utils import Grad3d
from Registration.spatial_transformation import SpatialTransformer, VecInt


class RegistrationModule(nn.Module):
    def __init__(self, model: nn.Module, inshape: [int, int, int]):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(size=inshape)
        self.model = model

    def forward(self, source: Tensor, target: Tensor) -> (Tensor, Tensor):
        x = torch.cat([source, target], dim=1)
        y = self.model(x)
        return y

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        return self.forward(source, target), self.forward(target, source)

    def registration_loss(self, source: tio.Subject, target: tio.Subject, forward_flow: Tensor, backward_flow: Tensor, sim_loss=nn.MSELoss(), num_classes : int = -1,lambda_sim : float = 1, lambda_seg : float = 1, lambda_mag : float = 1, lambda_grad : float = 1,  device='cuda'):
        target_image = target['image'][tio.DATA].to(device)
        source_image = source['image'][tio.DATA].to(device)
        target_label = target['label'][tio.DATA].float().to(device).float()
        source_label = source['label'][tio.DATA].float().to(device).float()

        if len(target_image.shape) == 4:
            target_image = target_image.unsqueeze(dim=0)
            target_label = target_label.unsqueeze(dim=0)
        if len(source_image.shape) == 4:
            source_image = source_image.unsqueeze(dim=0)
            source_label = source_label.unsqueeze(dim=0)

        loss_pair = torch.zeros((4)).to(device).float()
        if lambda_sim > 0:
            loss_pair[0] = lambda_sim * (sim_loss(target_image, self.warp(source_image, forward_flow)) + sim_loss(source_image, self.warp(target_image, backward_flow)))
        if lambda_seg > 0:
            warped_source_label = self.warp(source_label, forward_flow, mode='nearest')
            warped_target_label = self.warp(target_label, backward_flow, mode='nearest')

            one_hot_warped_source = F.one_hot(warped_source_label.squeeze(0).long(), num_classes=num_classes).float()
            one_hot_warped_target = F.one_hot(warped_target_label.squeeze(0).long(), num_classes=num_classes).float()
            one_hot_source = F.one_hot(source_label.squeeze(0).long(), num_classes=num_classes).float()
            one_hot_target = F.one_hot(target_label.squeeze(0).long(), num_classes=num_classes).float()

            loss_pair[1] = lambda_seg * (F.mse_loss(one_hot_target[..., 1::], one_hot_warped_source[..., 1::]) + F.mse_loss(one_hot_source[..., 1::], one_hot_warped_target[..., 1::]))
        if lambda_mag > 0:
            loss_pair[2] = lambda_mag * (F.mse_loss(torch.zeros(forward_flow.shape, device=device), forward_flow) + F.mse_loss(torch.zeros(backward_flow.shape, device=device), backward_flow))
        if lambda_grad > 0:
            loss_pair[3] = lambda_grad * (Grad3d().forward(forward_flow) + Grad3d().forward(backward_flow))
        return loss_pair

    def warp(self, tensor: Tensor, flow: Tensor, mode='bilinear') -> Tensor:
        return self.spatial_transformer(tensor, flow, mode=mode)

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


class RegistrationModuleSVF(RegistrationModule):
    def __init__(self, model: nn.Module, inshape: [int, int, int], int_steps: int = 7):
        super().__init__(model=model, inshape=inshape)
        self.vecint = VecInt(inshape=inshape, nsteps=int_steps)

    def velocity_to_flow(self, velocity):
        forward_flow = self.vecint(velocity)
        backward_flow = self.vecint(-velocity)
        return forward_flow, backward_flow

    def load_network(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        velocity = self.forward(source, target)
        forward_flow, backward_flow = self.velocity_to_flow(velocity)
        return forward_flow, backward_flow
