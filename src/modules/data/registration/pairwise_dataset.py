import torchio as tio
from typing import Sequence
import random
import itertools


class PairwiseSubjectsDataset(tio.SubjectsDataset):
    '''
        TorchIO SubjectsDataset that return a pair of subjects within all possible combinaisons.
        The dataset length is the number of possible combinaisons.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__(subjects, transform=transform)
        combinations = list(itertools.combinations(range(len(subjects)), 2))
        self.all_pair = combinations + [(j, i) for i, j in combinations]

    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.all_pair)

    def __getitem__(self, idx: int) -> dict[str, tio.Subject]:
        '''
            Get the pair of subjects at index idx
            :param idx: index of the pair of subjects
        '''
        subject_pair = {
            str(0): tio.SubjectsDataset.__getitem__(self, self.all_pair[idx][0]),
            str(1): tio.SubjectsDataset.__getitem__(self, self.all_pair[idx][1])
        }
        return subject_pair


class RandomPairwiseSubjectsDataset(tio.SubjectsDataset):
    '''
        Torch subjects_dataset that return a random pair of subjects within all possible combinaisons
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose):
        '''
        RandomPairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__(subjects, transform=transform)

    def __len__(self) -> int:
        '''
            Return the number of subjects in the dataset
        '''
        return len(self._subjects)

    def __getitem__(self, idx: int) -> dict[str, tio.Subject]:
        '''
            Get a random pair of subjects for the subject at index idx.
            :param idx: index of the pair of subjects
        '''
        while (rand := random.randint(0, len(self._subjects) - 1)) == idx:
            pass
        subject_pair = {
            str(0): tio.SubjectsDataset.__getitem__(self, idx),
            str(1): tio.SubjectsDataset.__getitem__(self, rand)
        }
        return subject_pair