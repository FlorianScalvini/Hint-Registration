import os
import json
import monai
import torch
import torchvision
import pandas as pd
from datetime import datetime
from typing import Callable, List
import torchio as tio
import datetime

def normalize_to_0_1(volume):
    '''
        Normalize volume to 0-1 range
    '''
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)


def subjects_from_csv(dataset_path: str, age=True, lambda_age: Callable = lambda x: x) -> List[tio.Subject]:
    """
    Function to create a list of subjects from a csv file
    Args:
        dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
        age: boolean to include age in the subject
        lambda_age: function to apply to the age
    """
    subjects = []
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        if age is False:
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label'])
            )
        else:
            age_value = lambda_age(row['age']) if lambda_age is not None else row['age']
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label']),
                age=age_value
            )
        subjects.append(subject)
    return subjects


def create_new_versioned_directory(base_name='', start_version=0):
    '''
        Create a new versioned directory
    '''
    # Check if version_0 exists
    version = start_version
    while os.path.exists(f'{base_name}_{version}'):
        version += 1
    new_version = f'{base_name}_{version}'
    os.makedirs(new_version)
    print(f'Created repository version: {new_version}')
    return new_version


def write_namespace_arguments(args, log_file='args_log.json'):
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args)
    }

    # If log file exists, append; else, create
    with open(log_file, 'w') as file:
        json.dump([log_entry], file, indent=4)


# Create a new directory recursively if it does not exist
def create_directory(directory):
    '''
        Create a new directory recursively if it does not exist
    '''
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
