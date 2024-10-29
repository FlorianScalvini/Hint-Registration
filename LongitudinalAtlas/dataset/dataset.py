import torch
import torchio as tio
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, t0: int, t1: int, transform: tio.Compose):
        subjects = []
        if t0 >= t1:
            raise ValueError("T0 must be lower than T1 !")

        df = pd.read_csv(dataset_path, sep=',')
        for index, row in df.iterrows():
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label']),
                age=(row["age"] - t0) / (t1 - t0)
            )
            subjects.append(subject)
        self.transform = transform
        self.subject_dataset = tio.SubjectsDataset(subjects, transform=transform)
        self.t0 = t0
        self.t1 = t1

    def __len__(self):
        return len(self.subject_dataset)

    def __getitem__(self, idx):
        indice_t0 = 0
        indice_t1 = 0
        for d in range(len(self.subject_dataset)):
            if self.subject_dataset[d]['age'] == 0:
                indice_t0 = d
            if self.subject_dataset[d]['age'] == 1:
                indice_t1 = d
        anchors = [indice_t0, idx, indice_t1]
        return {str(i): self.subject_dataset[anchors[i]] for i in range(3)}


