import itertools
import pandas as pd
import torchio as tio
from dataset.pairwise_subject import PairwiseSubjectsDataset
from dataset.subject_dataset import subjects_from_csv
from torch.utils.data import Dataset, DataLoader


# Dataset that return a triplet of subjects within a dict where the first and the second subjects are T0 / T1 subjects
class TripletStaticAnchorsDataset(tio.SubjectsDataset):
    def __init__(self, dataset_path: str, t0: int, t1: int, transform: tio.Compose):
        if t0 >= t1:
            raise ValueError("T0 must be lower than T1 !")
        subjects = subjects_from_csv(dataset_path=dataset_path, age=True, lambda_age=lambda x: (x - t0) / (t1 - t0))
        super().__init__(subjects, transform)
        self.t0 = t0
        self.t1 = t1
        self.subject_0 = None
        self.subject_1 = None
        for d in range(super().__len__()):
            if self._subjects[d]['age'] == 0:
                self.subject_0 = super().__getitem__(d)
            if self._subjects[d]['age'] == 1:
                self.subject_1 = super().__getitem__(d)

    def __len__(self) -> int:
        return super().__len__() - 2

    def __getitem__(self, idx) -> dict:
        anchors = [self.subject_0, self.subject_1, super().__getitem__(idx + 1)]
        return {str(i): anchors[i] for i in range(3)}


# Dataset that return a triplet of subjects within a dict within the all possible comb
class TripletSubjectDataset(PairwiseSubjectsDataset):
    def __init__(self, dataset_path: str, t0: int, t1: int, transform: tio.Compose):
        if t0 >= t1:
            raise ValueError("T0 must be lower than T1 !")
        super().__init__(dataset_path=dataset_path, transform=transform, num_elem=3, age=True, lambda_age=lambda x: (x - t0) / (t1 - t0))
        self.t0_idx = None
        self.t1_idx = None
        for d in range(super().__len__()):
            if self._subjects[d]['age'] == 0:
                self.t0_idx = d
            if self._subjects[d]['age'] == 1:
                self.t1_idx = d

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx) -> dict:
        return super().__getitem__(idx=idx)


