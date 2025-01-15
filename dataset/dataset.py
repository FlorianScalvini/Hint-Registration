import torchio as tio
import pandas as pd


class CustomDataset(tio.SubjectsDataset):
    def __init__(self, csv_path, transform=None):
        subjects = []
        self.transform = transform
        df = pd.read_csv(csv_path, sep=',')
        for index, row in df.iterrows():
            subjects.append(
                tio.Subject(
                    image=tio.ScalarImage(row["image"]),
                    label=tio.LabelMap(row["label"])
                )
            )
        super().__init__(subjects, transform)

    def __getitem__(self, index):
        return super().__getitem__(index=index)

    def __len__(self):
        return super().__len__()
