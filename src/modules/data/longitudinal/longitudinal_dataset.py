import torchio as tio
from typing import Sequence


class LongitudinalSubjectDataset(tio.SubjectsDataset):
    '''
        LongitudinalSubjectDataset is a subclass of torchio.SubjectsDataset with a fixed length of 1.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose):
        """
        :param subjects:
        :param transform:
        """
        super().__init__(subjects, transform=transform)
        self.num_subjects = len(subjects)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx) -> tio.Subject:
        return tio.SubjectsDataset.__getitem__(self, idx)
