import torch
import torch.nn as nn
from torch import Tensor
import monai
import torch.nn.functional as F
from .registration import  RegistrationModule

class PairwiseRegistrationModuleVelocity(RegistrationModule):
    '''
        Registration module for 3D image registration.yaml with stationary velocity field
        based on the DVF Registration module
    '''
    def __init__(self, model: nn.Module, int_steps: int = 7):
        '''
        :param model: nn.Module
        :param int_steps: int
        '''
        super().__init__()
        self.model = model
        self.dvf2ddf = monai.networks.blocks.DVF2DDF(num_steps=int_steps, mode='bilinear', padding_mode='zeros')# Vector integration based on Runge-Kutta method

    def forward(self, data: Tensor) -> Tensor:
        '''
            Forward pass of the registration module
            :param data: Input images
            :return: Deformation field
        '''
        return self.model(data)

    def velocity2displacement(self, dvf: Tensor) -> Tensor:
        '''
            Convert the velocity field to a flow field
            :param dvf: Velocity field
            :return: Deformation field
        '''
        return self.dvf2ddf(dvf)


