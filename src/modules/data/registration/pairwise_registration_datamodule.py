import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torchvision import transforms
import pandas as pd
import random
from .pairwise_dataset import RandomPairwiseSubjectsDataset, PairwiseSubjectsDataset

class PairwiseRegistrationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str,
                 rsize: int | tuple[int, int, int],
                 csize: int | tuple[int, int, int],
                 batch_size: int = 1,
                 num_workers: int = 15,
                 seed: int = None):
        """
        Data module for registration task.
        :param data_dir: Path to the CSV file containing image and label paths.
        :param rsize: Resize the image to this size.
        :param csize: Crop or pad the image to this size before resizing.
        :param batch_size: Batch size for training.
        :param num_workers: Number of workers for data loading.
        :param seed: Random seed for shuffling data.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rsize = rsize
        self.csize = csize
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.seed = seed

    def prepare_data(self):
        """Download or prepare data (if needed)."""
        pass

    def setup(self, stage=None):
        subjects = []
        df = pd.read_csv(self.data_dir)
        for index, row in df.iterrows():
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label'])
            )
            subjects.append(subject)
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(subjects)
        total_size = len(subjects)
        train_size = int(0.8 * total_size)
        val_size = int(0.2 * total_size)
        self.train_subjects = subjects
        self.val_subjects = subjects
        self.test_subjects = subjects

    def train_dataloader(self) -> tio.SubjectsLoader:
        transform = transforms.Compose([
            tio.CropOrPad(self.csize),
            tio.Resize(self.rsize),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
            tio.OneHot(),
        ])
        train_dataset = RandomPairwiseSubjectsDataset(self.train_subjects, transform=transform)
        return tio.SubjectsLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        transform = transforms.Compose([
            tio.CropOrPad(self.csize),
            tio.Resize(self.rsize),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
        ])
        val_dataset = PairwiseSubjectsDataset(self.val_subjects, transform=transform)
        return tio.SubjectsLoader(val_dataset,
                                  batch_size=1,
                                  num_workers=self.num_workers,
                                  persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        transform = transforms.Compose([
            tio.CropOrPad(self.csize),
            tio.Resize(self.rsize),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
        ])
        test_dataset = PairwiseSubjectsDataset(self.test_subjects, transform=transform)
        return tio.SubjectsLoader(test_dataset, batch_size=1,
                                  num_workers=self.num_workers,
                                  persistent_workers=True)