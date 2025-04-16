import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def animate_flow_3d(tensors : torch.Tensor, tensors_flow : torch.Tensor, axis: int, save_path: str):
    os.makedirs('evals/tmp', exist_ok=True)
    timesteps = tensors.shape[0]
    for index in range(timesteps):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4), tight_layout=True)
        ax[0].set_title('Fixed Image')
        ax[0].imshow(tensors[:, :, :, :, mid_frame_index].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[0].axis('off')
        ax[1].set_title('Moving Image')
        ax[1].imshow(Jw[:, :, :, :, mid_frame_index].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[1].axis('off')
        xy = torch.cat([xyz[:, :, :, mid_frame_index, 1].unsqueeze(-1), xyz[:, :, :, mid_frame_index, 2].unsqueeze(-1)],
                       dim=-1)
        xy = xy[:, ::down_factor, ::down_factor, :]
        segs1 = xy[0].cpu().numpy()
        segs2 = segs1.transpose(1, 0, 2)
        ax[2].add_collection(LineCollection(segs1))
        ax[2].add_collection(LineCollection(segs2))
        ax[2].autoscale()
        ax[2].set_title('Deformation Grid')
        ax[2].axis('off')
        fig.savefig(f'evals/tmp/forward_{index}.png')
        plt.close()

        fig, ax = plt.subplots(1, 3, figsize=(16, 4), tight_layout=True)
        ax[0].set_title('Fixed Image')
        ax[0].imshow(moving_img[:, :, :, :, mid_frame_index].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[0].axis('off')
        ax[1].set_title('Moving Image')
        ax[1].imshow(Iw[:, :, :, :, mid_frame_index].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax[1].axis('off')
        xyr = torch.cat(
            [xyzr[:, :, :, mid_frame_index, 1].unsqueeze(-1), xyzr[:, :, :, mid_frame_index, 2].unsqueeze(-1)], dim=-1)
        xyr = xyr[:, ::down_factor, ::down_factor, :]
        segs1 = xyr[0].cpu().numpy()
        segs2 = segs1.transpose(1, 0, 2)
        ax[2].add_collection(LineCollection(segs1))
        ax[2].add_collection(LineCollection(segs2))
        ax[2].autoscale()
        ax[2].set_title('Deformation Grid')
        ax[2].axis('off')
        fig.savefig(f'evals/tmp/reverse_{index}.png')
        plt.close()

    f_images = []
    r_images = []
    for index in range(num_frames):
        f_img = io.imread(f'evals/tmp/forward_{index}.png').astype(np.float32) / 255.
        r_img = io.imread(f'evals/tmp/reverse_{index}.png').astype(np.float32) / 255.
        f_images.append(f_img)
        r_images.append(r_img)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    im = ax.imshow(f_images[0], cmap='gray', animated=True)

    def update(i):
        im.set_array(f_images[i])

        return im

    animation_fig = animation.FuncAnimation(fig, update, frames=num_frames, repeat_delay=10)
    filename = 'forward' if prefix is None else f'{prefix}_forward'
    animation_fig.save(f'{save_path}/{filename}.gif')

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    im = ax.imshow(r_images[0], cmap='gray', animated=True)

    def update(i):
        im.set_array(r_images[i])

        return im

    animation_fig = animation.FuncAnimation(fig, update, frames=num_frames, repeat_delay=10)
    filename = 'reverse' if prefix is None else f'{prefix}_reverse'
    animation_fig.save(f'{save_path}/{filename}.gif')
    plt.close()
    shutil.rmtree('evals/tmp')




