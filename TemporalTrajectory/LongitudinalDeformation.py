import torch
import itertools
import numpy as np
import torch.nn as nn
import torchio2 as tio
from Registration import RegistrationModuleSVF
from TemporalTrajectory.network import MLP

class LongitudinalDeformation(torch.nn.Module):
    def __init__(self, t0, t1):
        super().__init__()
        self.t0 = t0
        self.t1 = t1

    def forward(self, data):
        return NotImplemented

    def getDeformationFieldFromTime(self, velocity: torch.Tensor, time: float):
        return self.reg_model.velocity_to_flow(velocity * time)

    def loads(self, path):
        return NotImplemented


class HadjHamouLongitudinalDeformation(LongitudinalDeformation):
    def __init__(self, reg_model : RegistrationModuleSVF, t0: int, t1: int, device: str):
        super().__init__(t0=t0, t1=t1)
        self.reg_model = reg_model
        self.device=device


    def forward(self, data: tio.SubjectsDataset):
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

    def loads(self, path):
        self.reg_model.load_state_dict(torch.load(path))



class OurLongitudinalDeformation(LongitudinalDeformation):

    def __init__(self, reg_model : RegistrationModuleSVF, mode: str, hidden_dim: int | None, num_layers: int | None, t0: int, t1: int):
        super().__init__(t0=t0, t1=t1)
        self.reg_model = reg_model
        self.mode = mode
        self.mlp_model = None
        if self.mode == 'mlp' and hidden_dim is not None:
            self.mlp_model = MLP(input_dim=1, output_dim=1, num_layers=num_layers, hidden_dim=hidden_dim)

    def forward(self, data : (torch.Tensor, torch.Tensor)):
        source, target = data
        if len(source.shape) == 4:
            source = source.unsqueeze(0)
        if len(target.shape) == 4:
            target = target.unsqueeze(0)
        velocity = self.reg_model.forward(source, target)
        return velocity

    def getDeformationFieldFrom2Times(self, velocity: torch.Tensor, time_a: float, time_b: float):
        if self.mode == 'mlp':
            time_a = torch.abs(self.mlp_model.forward(torch.asarray([time_a]).to(velocity.device)))
            time_b = torch.abs(self.mlp_model.forward(torch.asarray([time_a]).to(velocity.device)))
        return self.reg_model.velocity_to_flow(velocity=velocity * (time_b - time_a))

    def getDeformationFieldFromTime(self, velocity: torch.Tensor, time: float):
        if self.mode == 'mlp':
            time = torch.abs(self.mlp_model.forward(torch.asarray([time]).to(velocity.device)))
        return self.reg_model.velocity_to_flow(velocity=velocity * time)

    def loads(self, path, path_mlp=None):
        state_dict = torch.load(path)
        self.reg_model.load_state_dict(state_dict)
        if self.mlp_model is not None and path_mlp is not None:
            self.mlp_model.load_state_dict(torch.load(path_mlp))


class OurLongitudinalDeformationINR(LongitudinalDeformation):
    def __init__(self, reg_model : RegistrationModuleSVF,  t0: int, t1: int, size : list[int], hidden_dim=32, num_layers=4, max_freq=10):
        super().__init__(t0=t0, t1=t1)
        self.reg_model = reg_model
        self.max_freq = max_freq
        self.temporal_model = MLP(input_dim=(len(size) + 1) * self.max_freq * 2, output_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
        self.size = size
        x_coords, y_coords, z_coords = torch.meshgrid(
            torch.linspace(0, self.size[0] - 1, self.size[0]),
            torch.linspace(0, self.size[1] - 1, self.size[1]),
            torch.linspace(0, self.size[2] - 1, self.size[2])
        )
        coords = torch.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], dim=-1)
        self.encoded_coords = self.positional_encoding_3d(coords, max_freq=self.max_freq)

    @staticmethod
    def positional_encoding_3d(positions, max_freq=10):
        encoded_positions = []
        for i in range(positions.shape[1]):
            freqs = 2 ** (np.arange(max_freq).astype(np.float32))
            encoded = []
            for freq in freqs:
                encoded.append(torch.sin(positions[:, i: i + 1] * freq))
                encoded.append(torch.cos(positions[:, i: i + 1] * freq))
            encoded_positions.append(torch.cat(encoded, dim=-1))
        return torch.cat(encoded_positions, dim=-1)

    def forward(self, data : (torch.Tensor, torch.Tensor)):
        source, target = data
        if len(source.shape) == 4:
            source = source.unsqueeze(0)
        if len(target.shape) == 4:
            target = target.unsqueeze(0)
        velocity = self.reg_model.forward(source, target)
        return velocity

    def getDeformationFieldFrom2Times(self, velocity: torch.Tensor, time_a: float, time_b: float):
        return NotImplemented

    def getDeformationFieldFromTime(self, velocity: torch.Tensor, time: float):
        device = next(self.reg_model.parameters()).device
        if(self.encoded_coords.device != device):
            self.encoded_coords = self.encoded_coords.to(device)
        t = self.positional_encoding_3d(torch.full((self.encoded_coords.shape[0], 1), time), max_freq=self.max_freq).to(device)
        encoded = torch.cat([self.encoded_coords, t], dim=-1)  # Combine (x, y, z, t)
        modulated_time = self.temporal_model.forward(encoded)
        modulated_time = modulated_time.view(velocity.shape[2], velocity.shape[3], velocity.shape[4], -1).permute(3, 0, 1, 2)
        return self.reg_model.velocity_to_flow(velocity=velocity * modulated_time)

    def loads(self, path, path_mlp=None):
        state_dict = torch.load(path)
        self.reg_model.load_state_dict(state_dict)
        if self.mlp_model is not None and path_mlp is not None:
            self.mlp_model.load_state_dict(torch.load(path_mlp))

