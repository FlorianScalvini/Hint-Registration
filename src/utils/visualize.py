

import torch
import matplotlib.pyplot as plt
from .grid_utils import _make_identity_grid

def save_determinant_intensity(deformed_grid: torch.Tensor) -> None:
    """
    This function receives the deformation grid output by FlowNet3D and plots the
    determinants of its Jacobian for better analysis.

    Args:
        deformed_grid (torch.Tensor): the deformation grid output by FlowNet3D with shape [1, D, H, W, 3]
        save_path (str): the directory where the plot is saved
        prefix (str): a prefix to the file name that is to be saved
    """
    dy = deformed_grid[:, 1:, :-1, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dx = deformed_grid[:, :-1, 1:, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dz = deformed_grid[:, :-1, :-1, 1:, :] - deformed_grid[:, :-1, :-1, :-1, :]


    det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    determinants = det0 - det1 + det2

    mid_size = determinants.size(3) // 2
    slice_det = determinants[0, :, :, mid_size].rot90()

    plt.imshow(slice_det, vmin=-2, vmax=0, cmap='gray')
    plt.colorbar(label='|J(x)|')  # This shows the colormap scale as a legend
    plt.show()

def compute_log_jacobian_det(u):
    # u: displacement field of shape (B, 3, D, H, W)
    # compute gradients along x, y, z
    du_dx = u[:, :, 2:, 1:-1, 1:-1] - u[:, :, :-2, 1:-1, 1:-1]
    du_dy = u[:, :, 1:-1, 2:, 1:-1] - u[:, :, 1:-1, :-2, 1:-1]
    du_dz = u[:, :, 1:-1, 1:-1, 2:] - u[:, :, 1:-1, 1:-1, :-2]
    du = torch.stack((du_dx, du_dy, du_dz), dim=-1)  # shape: (B, 3, D-2, H-2, W-2, 3)

    # Identity + gradient = Jacobian
    J = du + torch.eye(3).to(u.device).view(1, 1, 1, 1, 1, 3, 3)
    # determinant
    detJ = torch.det(J)
    return detJ


def plt_grid(xy: torch.Tensor, factor, **kwargs):
    """
    Plots the 2D grid
    Args:
        xy (torch.Tensor): generated grids [h, w]
    """
    xy = xy[::factor, ::factor, :]
    H, W, _ = xy.shape

    # Downsample the grid by the given factor (to avoid overplotting)
    u_downsampled = xy[::factor, ::factor, :]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot horizontal lines (connecting points row-wise)
    for i in range(u_downsampled.shape[0]):
        ax.plot(u_downsampled[i, :, 0], u_downsampled[i, :, 1], 'b-', lw=1)  # Line for each row

    # Plot vertical lines (connecting points column-wise)
    for j in range(u_downsampled.shape[1]):
        ax.plot(u_downsampled[:, j, 0], u_downsampled[:, j, 1], 'b-', lw=1)  # Line for each column

    ax.set_title('2D Coordinate Grid with Lines')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.set_aspect('equal', 'box')  # Keep the aspect ratio equal
    plt.tight_layout()
    return fig

