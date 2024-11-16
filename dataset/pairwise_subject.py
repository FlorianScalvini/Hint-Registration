import torch.utils.data
import torchio as tio
from dataset.subject_dataset import subjects_from_csv
from typing import Callable, List
import itertools


class PairwiseSubjectsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, transform: tio.Compose, num_elem=2, age=True, lambda_age: Callable = lambda x: x):
        super().__init__()
        subjects = subjects_from_csv(dataset_path=dataset_path, age=age, lambda_age=lambda_age)
        self.pair_combinaisons = list(itertools.combinations(range(len(subjects)), num_elem))
        self.num_elem = num_elem
        self.dataset = tio.SubjectsDataset(subjects, transform=transform)

    def __len__(self):
        return len(self.pair_combinaisons)

    def __getitem__(self, idx):
        pair_index = self.pair_combinaisons[idx]
        return {str(i): self.dataset[pair_index[i]] for i in range(self.num_elem)}

