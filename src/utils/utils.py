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


# Create a new directory recursively if it does not exist
def create_directory(directory):
    '''
        Create a new directory recursively if it does not exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
    return directory



