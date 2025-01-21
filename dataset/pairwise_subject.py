import torch.utils.data
import torchio2 as tio
from .subject_dataset import subjects_from_csv
from typing import Callable, List
import itertools
import random

class PairwiseSubjectsDataset(tio.SubjectsDataset):
    '''
        Torch dataset that return a pair of subjects within all possible combinaisons
    '''
    def __init__(self, dataset_path: str, transform: tio.Compose, num_elem=2, age=True, lambda_age: Callable = lambda x: x):
        subjects = subjects_from_csv(dataset_path=dataset_path, age=age, lambda_age=lambda_age)
        super().__init__(subjects, transform=transform)
        self.pair_combinaisons = list(itertools.combinations(range(len(subjects)), num_elem))
        self.num_elem = num_elem

    def __len__(self):
        return len(self.pair_combinaisons)

    def __getitem__(self, idx):
        pair_index = self.pair_combinaisons[idx]
        return {str(i): tio.SubjectsDataset.__getitem__(self, pair_index[i]) for i in range(self.num_elem)}



class RandomPairwiseSubjectsDataset(tio.SubjectsDataset):
    '''
        Torch dataset that return a random pair of subjects within all possible combinaisons
    '''
    def __init__(self, dataset_path: str, transform: tio.Compose, num_elem=2, age=True, lambda_age: Callable = lambda x: x):
        subjects = subjects_from_csv(dataset_path=dataset_path, age=age, lambda_age=lambda_age)
        super().__init__(subjects, transform=transform)
        self.pair_combinaisons = list(itertools.combinations(range(len(subjects)), num_elem))
        self.num_elem = num_elem

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        pair_index = self.pair_combinaisons[random.randint(0, len(self.pair_combinaisons) - 1)]
        return {str(i): tio.SubjectsDataset.__getitem__(self, pair_index[i]) for i in range(self.num_elem)}

