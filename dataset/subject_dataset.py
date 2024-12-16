import torch
import random
import pandas as pd
import torchio as tio
from typing import Callable, List


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


class RandomSubjectsDataset(torch.utils.data.Dataset):
    '''
       Torch dataset that return N random subjects
       Args:
              dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
              transform: transformation to apply to the data
              lambda_age: function to apply to the age
              number_elem: number N of elements to return
    '''
    def __init__(self, dataset_path: str, transform: tio.Compose, lambda_age=None, number_elem=5):
        if lambda_age is None:
            lambda_age = lambda x: x
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda_age)
        self.dataset = tio.SubjectsDataset(subjects, transform=transform)
        self.num_elem = number_elem
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx) -> dict:
        sample = random.sample(range(0,len(self.dataset)), self.num_elem)
        return {str(i): self.dataset[sample[i]] for i in range(self.num_elem)}


class WrappedSubjectDataset(torch.utils.data.Dataset):
    '''
        Torch dataset that return a subject
        Args:
            dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
            transform: transformation to apply to the data
            lambda_age: function to apply to the age
    '''
    def __init__(self, dataset_path: str, transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda_age)
        self.dataset = tio.SubjectsDataset(subjects, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        return self.dataset[idx]

class OneWrappedSubjectDataset(torch.utils.data.Dataset):
    '''
        Torch dataset that return a subject with dataset size = 1
        Args:
            dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
            transform: transformation to apply to the data
            lambda_age: function to apply to the age
    '''
    def __init__(self, dataset_path: str, transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda_age)
        self.dataset = tio.SubjectsDataset(subjects, transform=transform)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx) -> dict:
        return self.dataset[idx]
