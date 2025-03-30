import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NCC(nn.Module):
    """
    Local (over window) normalized cross-correlation loss.
    """

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, x_pred, x_true):
        assert x_pred.shape == x_true.shape, "The shape of image I and J should be the same."

        # Get dimension of volume
        ndims = len(x_pred.shape) - 2
        assert ndims in [1, 2, 3], f"Volumes should be 1 to 3 dimensions, found: {ndims}"

        # Set window size
        win = [9] * ndims if self.win is None else self.win

        # Compute filters
        sum_filt = torch.ones([1, x_pred.shape[1], *win], device=x_pred.device)

        pad_no = win[0] // 2

        if ndims == 1:
            stride, padding = (1,), (pad_no,)
        elif ndims == 2:
            stride, padding = (1, 1), (pad_no, pad_no)
        else:
            stride, padding = (1, 1, 1), (pad_no, pad_no, pad_no)

        # Get convolution function
        conv_fn = getattr(F, f'conv{ndims}d')

        # Compute CC squares
        y_pred = x_pred * x_pred
        y_true = x_true * x_true
        y = x_pred * x_true

        x_pred_sum = conv_fn(x_pred, sum_filt, stride=stride, padding=padding)
        x_true_sum = conv_fn(x_true, sum_filt, stride=stride, padding=padding)
        y_pred_sum = conv_fn(y_pred, sum_filt, stride=stride, padding=padding)
        y_true_sum = conv_fn(y_true, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = x_pred_sum / win_size
        u_J = x_true_sum / win_size

        cross = y_sum - u_J * x_pred_sum - u_I * x_true_sum + u_I * u_J * win_size
        I_var = y_pred_sum - 2 * u_I * x_pred_sum + u_I * u_I * win_size
        J_var = y_true_sum - 2 * u_J * x_true_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
