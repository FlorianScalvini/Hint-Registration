import monai
import torch
import torch.nn as nn
from torch import Tensor
from .similarity import NCCLoss
from .jacobian import Jacobianloss
import torch.nn.functional as F

def GetLoss(loss):
    '''
    Get loss function from string

    Args:
    loss (str): The name of the loss function to retrieve. Possible values are:
      - 'mse': Mean Squared Error loss.
      - 'mae': Mean Absolute Error loss.
      - 'ncc': Normalized Cross Correlation loss.
      - 'lncc': Local Normalized Cross Correlation loss (monai).
      - 'dice': Dice loss (monai).
      - 'dicece': Dice Cross Entropy loss (monai).
      - 'mi': Global Mutual Information loss (monai).

    Returns:
    torch.nn.modules.loss._Loss: The loss function corresponding to the given name.

    Raises:
    ValueError: If the given loss function name is unknown.
    '''
    # Check the value of the loss parameter and return the corresponding loss function

    if loss == 'mse':
        return nn.MSELoss()
    elif loss == 'mae':
        return nn.L1Loss()
    elif loss == 'ncc':
        return NCCLoss()
    elif loss == 'lncc':
        return monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21, kernel_type='rectangular', reduction="mean")
    elif loss == 'dice':
        return monai.losses.DiceLoss()
    elif loss == 'dicece':
        return monai.losses.DiceCELoss()
    elif loss == 'mi':
        return monai.losses.GlobalMutualInformationLoss(num_bins=32)
    elif loss == 'be':
        return monai.losses.BendingEnergyLoss()
    elif loss == 'diff':
        return monai.losses.DiffusionLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss}")



class PairwiseRegistrationLoss(nn.Module):
    def __init__(self, sim_loss: nn.Module = None, seg_loss: nn.Module = None, mag_loss: nn.Module= None,
                 grad_loss: nn.Module= None, inv_loss: nn.Module = None, lambda_sim: float = 0, lambda_seg: float = 0,
                 lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__()
        self.sim_loss = GetLoss(loss=sim_loss)
        self.grad_loss = grad_loss
        self.mag_loss = mag_loss
        self.seg_loss = seg_loss
        self.inv_loss = inv_loss
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor, source_label: torch.Tensor,
                target_label: torch.Tensor, f: Tensor, bf: Tensor, v: Tensor = None) -> Tensor:
        '''
        Compute the registration loss for a pair of subjects.
        :param source_image: Source image
        :param target_image: Target image
        :param source_label: Source label
        :param target_label: Target label
        :param f: Flow field
        :param bf: Backward flow field
        :param v: Velocity field (Set to None to penalize the displacement field)
        '''
        loss_errors = torch.zeros(4).float().to(source_image.device)
        # Similarity loss
        if self.lambda_sim > 0:
            loss_errors[0] = self.lambda_sim * self.sim_loss(source_image, target_image)

        # Segmentation loss
        if self.lambda_seg > 0:
            loss_errors[1] = self.lambda_seg * F.mse_loss(source_label, target_label, reduction='none').mean(dim=(0, 2, 3, 4)).sum()

        # Magnitude loss
        if self.lambda_mag > 0:
            loss_errors[2] = self.lambda_mag * (self.mag_loss(f) if v is None else self.mag_loss(v))

        # Gradient loss
        if self.lambda_grad > 0:
            loss_errors[3] = self.lambda_grad * (self.grad_loss(f) if v is None else self.grad_loss(v))

        return loss_errors

