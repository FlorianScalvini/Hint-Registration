import torch
import random
import itertools
import pandas as pd
import torchio as tio
from dataset.pairwise_subject import PairwiseSubjectsDataset
from dataset.subject_dataset import subjects_from_csv
from torch.utils.data import Dataset, DataLoader


# Dataset that return a triplet of subjects within a dict where the first and the second subjects are T0 / T1 subjects
class TripletStaticAnchorsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, t0: int, t1: int, transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda_age)
        self.dataset = tio.SubjectsDataset(subjects=subjects, transform=transform)
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.subject_0 = self.dataset[2]
        self.subject_1 = self.dataset[10]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        anchors = [self.subject_0, self.subject_1, self.dataset[idx]]
        return {str(i): anchors[i] for i in range(3)}


# Dataset that return a triplet of subjects within a dict within the all possible comb
class TripletSubjectDataset(PairwiseSubjectsDataset):
    def __init__(self, dataset_path: str,  transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        super().__init__(dataset_path=dataset_path, transform=transform, num_elem=3, age=True, lambda_age=lambda_age)


    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx) -> dict:
        return super().__getitem__(idx=idx)


class RandomTripletSubjectDataset(torch.utils.data.Dataset):
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


class RandomSubjectsDataset(torch.utils.data.Dataset):
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
    def __init__(self, dataset_path: str, transform: tio.Compose, lambda_age=None):
        if lambda_age is None:
            lambda_age = lambda x: x
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda_age)
        self.dataset = tio.SubjectsDataset(subjects, transform=transform)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx) -> dict:
        return self.dataset[idx]
