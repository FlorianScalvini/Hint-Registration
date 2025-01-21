import torch
import itertools
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
        self.velocity = None

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

    def __init__(self, reg_model : RegistrationModuleSVF, mode: str, hidden_mlp_layer: list[int] | None, t0: int, t1: int):
        super().__init__(t0=t0, t1=t1)
        self.reg_model = reg_model
        self.mode = mode
        self.mlp_model = None
        if self.mode == 'mlp' and hidden_mlp_layer is not None:
            self.mlp_model = MLP(hidden_size=hidden_mlp_layer)

    def forward(self, data : (torch.Tensor, torch.Tensor)):
        source, target = data
        if len(source.shape) == 4:
            source = source.unsqueeze(0)
        if len(target.shape) == 4:
            target = target.unsqueeze(0)
        velocity = self.reg_model.forward(source, target)
        return velocity

    def getDeformationFieldFrom2Times(self, velocity: torch.Tensor, time_a: float, time_b: float):
        if 'mode' == 'mlp':
            time_a = torch.abs(self.mlp_model.forward(torch.asarray([time_a])))
            time_b = torch.abs(self.mlp_model.forward(torch.asarray([time_a])))
        return self.reg_model.velocity_to_flow(velocity=velocity * (time_b - time_a))


    def getDeformationFieldFromTime(self, velocity: torch.Tensor, time: float):
        if 'mode' == 'mlp':
            time = torch.abs(self.mlp_model.forward(torch.asarray([time])))
        return self.reg_model.velocity_to_flow(velocity=velocity * time)

    def loads(self, path, path_mlp=None):
        state_dict = torch.load(path)
        self.reg_model.load_state_dict(state_dict)
        if self.mlp_model is not None and path_mlp is not None:
            self.mlp_model.load_state_dict(torch.load(path_mlp))

