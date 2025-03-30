import torch
import itertools
import os

from .registration import RegistrationModuleSVF
from .blocks.mlp import MLP
from .blocks.inr import ImplicitNeuralNetwork
import torchio as tio


class LongitudinalDeformation(torch.nn.Module):
    def __init__(self, t0, t1):
        super().__init__()
        self.t0 = t0
        self.t1 = t1

    def forward(self, data) -> torch.Tensor:
        return NotImplemented

    def deformation_from_time(self, velocity: torch.Tensor, time: float) -> torch.Tensor:
        return NotImplemented

    def load_model(self, path) -> None:
        return


class HadjHamouLongitudinalDeformation(LongitudinalDeformation):
    '''
    Implementation of Hadj-Hamou longitudinal deformation model
    '''
    def __init__(self, reg_model : RegistrationModuleSVF, t0: int, t1: int):
        '''
        Hadj-Hamou longitudinal deformation model
        :param model: Registration model
        :param t0: time 0
        :param t1: time 1
        '''
        super().__init__(t0=t0, t1=t1)
        self.reg_model = model

    def forward(self, data: tio.SubjectsDataset) -> torch.Tensor:
        denum = 0
        in_shape = data[0]['image'][tio.DATA].shape[1:]
        num = torch.zeros([1, 3] + list(in_shape)).to(self.device)
        transformationPairs = list(itertools.combinations(range(len(data)), 2))
        with torch.no_grad():
            for i, j in transformationPairs:
                sample_i = data[i]
                sample_j = data[j]
                time_ij = sample_j['age'] - sample_i['age']
                velocity_ij = self.reg_model(sample_i['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device),
                                             sample_j['image'][tio.DATA].unsqueeze(dim=0).float().to(self.device))
                num += velocity_ij * time_ij
                denum += time_ij * time_ij
            velocity = num / denum if denum != 0 else torch.zeros_like(num)
        return velocity

    def deformation_from_time(self, velocity: torch.Tensor, time: float):
        return self.reg_model.velocity_to_flow(velocity * time)

    def load_model(self, path):
        self.reg_model.load_state_dict(torch.load(path))


class OurLongitudinalDeformation(LongitudinalDeformation):

    def __init__(self, reg_model : RegistrationModuleSVF, mode: str, t0: int, t1: int, hidden_dim: int | None = 32,
                 num_layers: int | None =4, size: list[int] | None = None, max_freq: int | None = 8):
        '''
        Our longitudinal deformation model
        :param reg_model: Registration model
        :param mode: Interpolation mode, either 'mlp', 'inr' or 'linear'
        :param t0: time 0
        :param t1: time 1
        :param hidden_dim: Hidden dimensions of the MLP or INR
        :param num_layers: Number of layers of the MLP or INR
        :param size: INR size
        :param max_freq: Maximum frequency of the INR
        '''
        super().__init__(t0=t0, t1=t1)
        self.reg_model = reg_model
        self.mode = mode
        self.temp_model = None
        if self.mode == 'mlp' and hidden_dim is not None:
            self.temp_model = MLP(input_dim=1, output_dim=1, num_layers=num_layers, hidden_dim=hidden_dim)
        if self.mode == 'inr' and hidden_dim is not None:
            self.max_freq = max_freq
            self.temp_model = ImplicitNeuralNetwork(size=size, hidden_dim=hidden_dim, num_layers=num_layers,
                                                    max_freq=max_freq)

    def forward(self, data : (torch.Tensor, torch.Tensor)) -> torch.Tensor:
        source, target = data
        if len(source.shape) == 4:
            source = source.unsqueeze(0)
        if len(target.shape) == 4:
            target = target.unsqueeze(0)
        velocity = self.reg_model.forward(source, target)
        return velocity

    def deformation_from_time(self, velocity: torch.Tensor, time: float) -> torch.Tensor:
        '''
        Compute the deformation from the velocity and time
        :param velocity: Velocity field
        :param time: Relative time
        :return: Deformation field
        '''
        if self.mode == 'mlp':
            time = self.temp_model(torch.asarray([time]).to(velocity.device))
        if self.mode == 'inr':
            time = self.temp_model(time)
        return self.reg_model.velocity_to_flow(velocity=velocity * time)

    def load_model(self, path) -> None:
        state_dict = torch.load(path)
        self.reg_model.load_state_dict(state_dict)

    def load_temporal(self, path) -> None:
        if self.temp_model is not None:
            self.temp_model.load_state_dict(torch.load(path))

    def save(self, path) -> None:
        torch.save(self.reg_model.state_dict(), os.path.join(path, 'model.pth'))
        if self.temp_model is not None:
            torch.save(self.temp_model.state_dict(), os.path.join(path, 'temporal_model.pth'))
