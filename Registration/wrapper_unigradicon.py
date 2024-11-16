import torch
from torch import Tensor
from unigradicon import get_unigradicon
from icon_registration.mermaidlite import compute_warped_image_multiNC

from Registration import RegistrationModule


class UniGradIconRegistrationWrapper(RegistrationModule):
    def __init__(self):
        super().__init__(model=get_unigradicon(), inshape=[175,175,175])

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        _ = self.model(source, target)
        return self.model.phi_AB_vectorfield, self.model.phi_BA_vectorfield

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        return self.forward(source, target)

    def wrap(self, tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return compute_warped_image_multiNC(tensor, flow, super().model.spacing, 0, zero_boundary=True)