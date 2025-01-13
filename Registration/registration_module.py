import torch
import torchio as tio
import torch.nn as nn
from utils import Grad3d
from torch import Tensor
import torch.nn.functional as F
from Registration.spatial_transformation import SpatialTransformer, VecInt
from monai.networks.blocks import Warp

class RegistrationModule(nn.Module):
    '''
        Registration module for 3D image registration with DVF
    '''
    def __init__(self, model: nn.Module, inshape: [int, int, int]):
        '''
        :param model: nn.Module
        :param inshape: [int, int, int]
        '''
        super().__init__()
        self.model = model
        self.spatial_transformer = SpatialTransformer(size=inshape)

    def forward(self, source: Tensor, target: Tensor) -> (Tensor, Tensor):
        '''
            Returns the deformation field from source to target image
        '''
        x = torch.cat([source, target], dim=1)
        y = self.model(x)
        return y

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor) -> (Tensor, Tensor):
        '''
        Returns the forward and backward flow
        '''
        return self.forward(source, target), self.forward(target, source)


    def registration_loss(self, source: tio.Subject, target: tio.Subject, forward_flow: Tensor, backward_flow: Tensor, sim_loss=nn.MSELoss(), lambda_sim : float = 1, lambda_seg : float = 1, lambda_mag : float = 1, lambda_grad : float = 1,  device='cuda') -> Tensor:
        '''
            Compute the registration loss for a pair of subjects
        '''
        loss_pair = torch.zeros((4)).float().to(device)
        target_image = target['image'][tio.DATA].to(device)
        source_image = source['image'][tio.DATA].to(device)

        if len(target_image.shape) == 4:
            target_image = target_image.unsqueeze(dim=0)
        if len(source_image.shape) == 4:
            source_image = source_image.unsqueeze(dim=0)

        if lambda_sim > 0:
            loss_pair[0] = sim_loss(target_image, self.warp(source_image, forward_flow)) + \
                           sim_loss(source_image, self.warp(target_image, backward_flow))
        if lambda_seg > 0:
            target_label = target['label'][tio.DATA].float().to(device).float()
            source_label = source['label'][tio.DATA].float().to(device).float()
            if len(target_label.shape) == 4:
                target_label = target_label.unsqueeze(dim=0)
            if len(source_label.shape) == 4:
                source_label = source_label.unsqueeze(dim=0)
            warped_source_label = self.warp(source_label.float(), forward_flow)
            warped_target_label = self.warp(target_label.float(), backward_flow)
            loss_pair[1] += F.mse_loss(target_label[:, 1::, ...].float(), warped_source_label[:, 1::, ...]) \
                            + F.mse_loss(source_label[:, 1::, ...].float(), warped_target_label[:, 1::, ...])
        if lambda_mag > 0:
            loss_pair[2] += (F.mse_loss(torch.zeros(forward_flow.shape, device=device), forward_flow) + F.mse_loss(torch.zeros(backward_flow.shape, device=device), backward_flow))

        if lambda_grad > 0:
            loss_pair[3] += self.regularizer(forward_flow, penalty='l2').to(device) + \
                            self.regularizer(backward_flow, penalty='l2').to(device)
        return loss_pair

    def warp(self, tensor: Tensor, flow: Tensor) -> Tensor:
        return self.spatial_transformer(tensor, flow)


    @staticmethod
    def regularizer(disp, penalty='l2') -> Tensor:
        '''
        Compute the regularizer of the displacement field
        '''
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
    '''
        Registration module for 3D image registration with stationary velocity field
        based on the DVF Registration module
    '''
    def __init__(self, model: nn.Module, inshape: [int, int, int], int_steps: int = 7):
        '''
        :param model: nn.Module
        :param inshape: [int, int, int]
        :param int_steps: int
        '''
        super().__init__(model=model, inshape=inshape)
        self.vecint = VecInt(inshape=inshape, nsteps=int_steps) # Vector integration based on Runge-Kutta method

    def velocity_to_flow(self, velocity):
        '''
            Convert the velocity field to a flow field (forward and backward)
        '''
        forward_flow = self.vecint(velocity)
        backward_flow = self.vecint(-velocity)
        return forward_flow, backward_flow

    def load_network(self, path):
        '''
            Load the network from a file
        '''
        self.model.load_state_dict(torch.load(path))

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        '''
            Returns the forward and backward flow after registration
            Override the parent method with SVF intergration between the Registration Network and the DVF output
        '''
        velocity = self.forward(source, target)
        forward_flow, backward_flow = self.velocity_to_flow(velocity)
        return forward_flow, backward_flow
