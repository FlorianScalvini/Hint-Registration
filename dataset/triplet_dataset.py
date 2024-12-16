import torch
import random
import itertools
import pandas as pd
import torchio as tio
from dataset.pairwise_subject import PairwiseSubjectsDataset
from dataset.subject_dataset import subjects_from_csv
from torch.utils.data import Dataset, DataLoader


class TripletStaticAnchorsDataset(torch.utils.data.Dataset):
    '''
        Torch dataset that return a triplet of subjects within a dict where the first and the second subjects are T0 / T1 subjects
        Args:
            dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
            t0: index of the first subject
            t1: index of the second subject
            transform: transformation to apply to the data
            lambda_age: function to apply to the age
    '''
    def __init__(self, dataset_path: str, t0: int, t1: int, transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda_age)
        self.dataset = tio.SubjectsDataset(subjects=subjects, transform=transform)
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.subject_0 = self.dataset[0]
        self.subject_1 = self.dataset[-1]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        anchors = [self.subject_0, self.subject_1, self.dataset[idx]]
        return {str(i): anchors[i] for i in range(3)}


class TripletSubjectDataset(PairwiseSubjectsDataset):
    '''
        Torch dataset that return a triplet of subjects within all possible combinaisons
        Args:
            dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
            transform: transformation to apply to the data
            lambda_age: function to apply to the age
    '''
    def __init__(self, dataset_path: str,  transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        super().__init__(dataset_path=dataset_path, transform=transform, num_elem=3, age=True, lambda_age=lambda_age)


    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx) -> dict:
        return super().__getitem__(idx=idx)


class RandomTripletSubjectDataset(torch.utils.data.Dataset):
    '''
        Torch dataset that return a random triplet of subjects within all possible combinaisons
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
        sample = random.sample(range(0,len(self.dataset)), 5)
        return {str(i): self.dataset[sample[i]] for i in range(5)}

