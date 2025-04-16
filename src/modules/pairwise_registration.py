import torch
import torch.nn as nn
from torch import Tensor
from monai import networks


class PairwiseRegistrationModule(nn.Module):
    '''
        Registration module for 3D image registration.yaml with DVF
    '''
    def __init__(self, model: nn.Module):
        '''
        :param model: nn.Module
        '''
        super().__init__()
        self.model = model # Registration model
        self.warp_block = networks.blocks.Warp(mode='bilinear', padding_mode='zeros', jitter=False) # Warping block for the registration module

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        '''
            Forward pass of the registration module
            :param source: Source image
            :param target: Target image
            :return: Deformation field
        '''
        return self.model(torch.cat([source, target], dim=1))

    def warp(self, tensor: Tensor, f: Tensor) -> Tensor:
        '''
            Warp an image by a flow field
            :param tensor: image
            :param f: Flow field
            :return: Warped image
        '''
        return self.warp_block(tensor, f)

    def load_network(self, path) -> None:
        '''
            Load the network weights
            :param path: Path to the weights
        '''
        self.model.load_state_dict(torch.load(path))


class PairwiseRegistrationModuleVelocity(PairwiseRegistrationModule):
    '''
        Registration module for 3D image registration.yaml with stationary velocity field
        based on the DVF Registration module
    '''
    def __init__(self, model: nn.Module, int_steps: int = 7):
        '''
        :param model: nn.Module
        :param int_steps: int
        '''
        super().__init__(model=model)
        self.dvf2ddf = networks.blocks.DVF2DDF(num_steps=int_steps, mode='bilinear', padding_mode='zeros')# Vector integration based on Runge-Kutta method

    def velocity2displacement(self, dvf: Tensor) -> Tensor:
        '''
            Convert the velocity field to a flow field
            :param dvf: Velocity field
            :return: Deformation field
        '''
        return self.dvf2ddf(dvf)


