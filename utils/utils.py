import os
import torch
import torchvision
from utils.transform import normalize_to_0_1
import monai

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

def get_activation_from_string(activation):
    if activation == 'ReLU':
        return torch.nn.ReLU
    elif activation == 'LeakyReLU':
        return torch.nn.LeakyReLU
    elif activation == 'Sigmoid':
        return torch.nn.Sigmoid
    elif activation == 'Tanh':
        return torch.nn.Tanh
    elif activation == 'Softmax':
        return torch.nn.Softmax
    elif activation == 'Softplus':
        return torch.nn.Softplus
    elif activation == 'Softsign':
        return torch.nn.Softsign
    elif activation == 'ELU':
        return torch.nn.ELU
    elif activation == 'SELU':
        return torch.nn.SELU
    elif activation == 'CELU':
        return torch.nn.CELU
    elif activation == 'GLU':
        return torch.nn.GLU
    elif activation == 'Hardshrink':
        return torch.nn.Hardshrink
    elif activation == 'Hardtanh':
        return torch.nn.Hardtanh
    elif activation == 'LogSigmoid':
        return torch.nn.LogSigmoid
    elif activation == 'Softmin':
        return torch.nn.Softmin
    elif activation == 'Softmax2d':
        return torch.nn.Softmax2d
    else:
        return None

def get_model_from_string(model_name):
    if model_name == 'Unet':
        return monai.networks.nets.Unet
    elif model_name == 'AttentionUnet':
        return monai.networks.nets.AttentionUnet
    else:
        return None