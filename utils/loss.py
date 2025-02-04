import math
import torch
import monai
import numpy as np
import torch.nn as nn
import torch.functional as F


class LNCC():
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        I = image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J sould be the same."

        return torch.mean(
            1
            - (self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (torch.relu(self.blur(I * I) - self.blur(I) ** 2) + 0.00001)
                * (torch.relu(self.blur(J * J) - self.blur(J) ** 2) + 0.00001)
            )
        )

def _get_gaussian_kernel1d(kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d

def gaussian_blur(tensor, kernel_size, sigma, padding="same"):
    kernel1d = _get_gaussian_kernel1d(kernel_size=kernel_size, sigma=sigma).to(
        tensor.device, dtype=tensor.dtype
    )
    out = tensor
    group = tensor.shape[1]

    if len(tensor.shape) - 2 == 1:
        out = torch.conv1d(out, kernel1d[None, None, :].expand(group,-1,-1), padding="same", groups=group)
    elif len(tensor.shape) - 2 == 2:
        out = torch.conv2d(out, kernel1d[None, None, :, None].expand(group,-1,-1,-1), padding="same", groups=group)
        out = torch.conv2d(out, kernel1d[None, None, None, :].expand(group,-1,-1,-1), padding="same", groups=group)
    elif len(tensor.shape) - 2 == 3:
        out = torch.conv3d(out, kernel1d[None, None, :, None, None].expand(group,-1,-1,-1,-1), padding="same", groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, :, None].expand(group,-1,-1,-1,-1), padding="same", groups=group)
        out = torch.conv3d(out, kernel1d[None, None, None, None, :].expand(group,-1,-1,-1,-1), padding="same", groups=group)

    return out



class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class BendingEnergyLoss(nn.Module):
    """
    regularization loss of bending energy of a 3d deformation field
    """

    def __init__(self, norm='L2', spacing=(1, 1, 1), normalize=True):
        super(BendingEnergyLoss, self).__init__()
        self.norm = norm
        self.spacing = torch.tensor(spacing).float()
        self.normalize = normalize
        if self.normalize:
            self.spacing /= self.spacing.min()

    def forward(self, input):
        """
        :param norm: 'L1' or 'L2'
        :param input: Nx3xDxHxW
        :return:
        """
        self.spacing = self.spacing.to(input.device)
        spatial_dims = torch.tensor(input.shape[2:]).float().to(input.device)
        if self.normalize:
            spatial_dims /= spatial_dims.min()

        # according to
        # f''(x) = [f(x+h) + f(x-h) - 2f(x)] / h^2
        # f_{x, y}(x, y) = [df(x+h, y+k) + df(x-h, y-k) - df(x+h, y-k) - df(x-h, y+k)] / 2hk

        ddx = torch.abs(input[:, :, 2:, 1:-1, 1:-1] + input[:, :, :-2, 1:-1, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1])\
                 .view(input.shape[0], input.shape[1], -1)

        ddy = torch.abs(input[:, :, 1:-1, 2:, 1:-1] + input[:, :, 1:-1, :-2, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]) \
                 .view(input.shape[0], input.shape[1], -1)

        ddz = torch.abs(input[:, :, 1:-1, 1:-1, 2:] + input[:, :, 1:-1, 1:-1, :-2] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]) \
                 .view(input.shape[0], input.shape[1], -1)

        dxdy = torch.abs(input[:, :, 2:, 2:, 1:-1] + input[:, :, :-2, :-2, 1:-1] -
                         input[:, :, 2:, :-2, 1:-1] - input[:, :, :-2, 2:, 1:-1]).view(input.shape[0], input.shape[1], -1)

        dydz = torch.abs(input[:, :, 1:-1, 2:, 2:] + input[:, :, 1:-1, :-2, :-2] -
                         input[:, :, 1:-1, 2:, :-2] - input[:, :, 1:-1, :-2, 2:]).view(input.shape[0], input.shape[1], -1)

        dxdz = torch.abs(input[:, :, 2:, 1:-1, 2:] + input[:, :, :-2, 1:-1, :-2] -
                         input[:, :, 2:, 1:-1, :-2] - input[:, :, :-2, 1:-1, 2:]).view(input.shape[0], input.shape[1], -1)


        if self.norm == 'L2':
            ddx = (ddx ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0]**2)) ** 2
            ddy = (ddy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1]**2)) ** 2
            ddz = (ddz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2]**2)) ** 2
            dxdy = (dxdy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0] * self.spacing[1])) ** 2
            dydz = (dydz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1] * self.spacing[2])) ** 2
            dxdz = (dxdz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2] * self.spacing[0])) ** 2

        d = (ddx.mean() + ddy.mean() + ddz.mean() + 2*dxdy.mean() + 2*dydz.mean() + 2*dxdz.mean()) / 9.0
        return d

class Grad3d:
    """
    3-D gradient loss.
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad

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
        return NCC()
    elif loss == 'lncc':
        return monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=15, smooth_dr=1e-8, kernel_type='rectangular', reduction="mean")
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