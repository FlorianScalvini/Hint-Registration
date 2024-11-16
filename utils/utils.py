import os
import torch
import torchvision
from utils.transform import normalize_to_0_1


# Create a new versioned directory
def create_new_versioned_directory(base_name='./version', start_version=0):
    # Check if version_0 exists
    version = start_version
    while os.path.exists(f'{base_name}_{version}'):
        version += 1
    new_version = f'{base_name}_{version}'
    os.makedirs(new_version)
    print(f'Created repository version: {new_version}')
    return new_version


# Create a new directory recursively if it does not exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
    return directory


def volume_to_batch_image(volume, normalize=True, dim='D', batch=0):
    """ Helper function that, given a 5 D tensor, converts it to a 4D
    tensor by choosing element batch, and moves the dim into the batch
    dimension, this then allows the slices to be tiled for tensorboard

    Args:
        volume: volume to be viewed

    Returns:
        3D tensor (already tiled)
    """
    if batch >= volume.shape[0]:
        raise ValueError('{} batch index too high'.format(batch))
    if dim == 'D':
        image = volume[batch, :, :, :, :].permute(1, 0, 2, 3)
    elif dim == 'H':
        image = volume[batch, :, :, :, :].permute(2, 0, 1, 3)
    elif dim == 'W':
        image = volume[batch, :, :, :, :].permute(3, 0, 1, 2)
    else:
        raise ValueError('{} dim not supported'.format(dim))
    if normalize:
        return torchvision.utils.make_grid(normalize_to_0_1(image))
    else:
        return torchvision.utils.make_grid(image)


def get_cuda_is_available_or_cpu():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")