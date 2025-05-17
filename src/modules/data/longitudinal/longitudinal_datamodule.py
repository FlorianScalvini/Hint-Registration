import pandas as pd
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torchvision import transforms
from .longitudinal_dataset import LongitudinalSubjectDataset


class LongitudinalDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 t0: int,
                 t1: int,
                 rsize: int | tuple[int, int, int],
                 csize: int | tuple[int, int, int],
                 batch_size: int = 1,
                 num_workers: int = 8,
                 num_classes: int = 20):
        """
        Data module for longitudinal registration task.
        :param data_dir: Path to the CSV file containing image and label paths.
        :param t0: Time point 0.
        :param t1: Time point 1.
        :param rsize: Resize the image to this size.
        :param csize: Crop or pad the image to this size before resizing.
        :param batch_size: Batch size for training.
        :param num_workers: Number of workers for data loading.
        :param num_classes: Number of classes for segmentation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.t0 = t0
        self.t1 = t1
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rsize = rsize
        self.csize = csize
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.seed = None
        self.num_classes = num_classes

    def prepare_data(self):
        """Download or prepare data (if needed)."""
        pass

    def setup(self, stage=None):
        subjects = []
        df = pd.read_csv(self.data_dir)
        for index, row in df.iterrows():
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label']),
                age=(row['age'] - self.t0) / (self.t1 - self.t0),
            )
            subjects.append(subject)
        subjects.sort(key=lambda x: x['age'])

        self.train_subjects = subjects
        self.val_subjects = subjects
        self.test_subjects = self.val_subjects

    def train_dataloader(self) -> tio.SubjectsLoader:
        transform = transforms.Compose([
            tio.CropOrPad(self.csize),
            tio.Resize(self.rsize),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
            tio.OneHot(self.num_classes),
        ])
        train_dataset =  LongitudinalSubjectDataset(self.train_subjects, transform=transform)
        return tio.SubjectsLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        transform = transforms.Compose([
            tio.CropOrPad(self.csize),
            tio.Resize(self.rsize),
        ])
        val_dataset = LongitudinalSubjectDataset(self.val_subjects, transform=transform)
        return tio.SubjectsLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        transform = transforms.Compose([
            tio.CropOrPad(self.csize),
            tio.Resize(self.rsize),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
        ])
        test_dataset = LongitudinalSubjectDataset(self.test_subjects, transform=transform)
        return tio.SubjectsLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)