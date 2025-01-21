import torch
import torch.nn as nn
from torch import Tensor
from Registration.spatial_transformation import SpatialTransformer, VecInt

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
