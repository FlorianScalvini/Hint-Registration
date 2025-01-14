import argparse

import numpy as np
import torch

def color_palette_20():
    return np.array([
            [0, 0, 0],       # Black
            [255, 255, 255],  # White
            [128, 128, 128],  # Bright Red
            [255, 0, 0],     # Bright Red
            [0, 255, 0],     # Bright Green
            [0, 0, 255],     # Bright Blue
            [255, 255, 0],   # Yellow
            [0, 255, 255],   # Cyan
            [255, 0, 255],   # Magenta
            [128, 0, 0],     # Dark Red
            [0, 128, 0],     # Dark Green
            [0, 0, 128],     # Dark Blue
            [128, 128, 0],   # Olive
            [128, 0, 128],   # Purple
            [0, 128, 128],   # Teal
            [192, 192, 192], # Light Gray
            [128, 128, 128], # Medium Gray
            [64, 64, 64],    # Dark Gray
            [255, 165, 0],   # Orange
            [75, 0, 130],    # Indigo
            [255, 192, 203], # Pink
            [173, 216, 230], # Light Blue
            [255, 255, 255],  # White

        ], dtype=np.uint8)

# Map labels to RGB colors
def map_labels_to_colors(label_tensor, color_palette=color_palette_20()):
    """
    Convert a 3D label tensor to an RGB tensor using a color palette.
    Args:
        label_tensor (torch.Tensor): 3D tensor of shape (D, H, W) with integer labels.
        color_palette (np.ndarray): Array of shape (num_classes, 3) with RGB values.
    Returns:
        torch.Tensor: 4D tensor of shape (3,D, H, W, 3) with RGB values.
    """
    label_numpy = label_tensor.numpy()
    rgb_image = color_palette[label_numpy]
    return torch.from_numpy(rgb_image)

def seg_map_error(pred_tensor, gt_tensor, dim=0):
    pred_tensor = pred_tensor.cpu().detach()
    gt_tensor = gt_tensor.cpu().detach()
    pred_tensor = torch.argmax(pred_tensor, dim=dim)
    gt_tensor = torch.argmax(gt_tensor, dim=dim)
    error_map = torch.zeros(pred_tensor.shape).int()
    error_map[pred_tensor != gt_tensor] = 1
    return error_map


