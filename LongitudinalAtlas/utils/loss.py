import math
import torch
import monai
import numpy as np
import torch.nn as nn
import torch.functional as F
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
        return monai.losses.LocalNormalizedCrossCorrelationLoss()
    elif loss == 'dice':
        return monai.losses.DiceLoss()
    elif loss == 'dicece':
        return monai.losses.DiceCELoss()
    elif loss == 'mi':
        return monai.losses.GlobalMutualInformationLoss(num_bins=32)
    else:
        raise ValueError(f"Unknown loss function: {loss}")