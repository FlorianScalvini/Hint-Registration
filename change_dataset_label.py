import argparse
import torch
import numpy
import pandas as pd
from dataset import subjects_from_csv
import torchio as tio

class_index = [
               (3, 1),
               (4, 1),
               (7, 2),
               (8, 2),]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuse and change label of dataset')
    parser.add_argument("--csv_path", type=str, default="./data/full_dataset.csv")
    parser.add_argument("--t0", type=float, default=21)
    parser.add_argument("--t1", type=float, default=36)
    arguments = parser.parse_args()

    df = pd.read_csv(arguments.csv_path)
    for index, row in df.iterrows():
        subject = tio.Subject(
            label=tio.LabelMap(row['label'])
        )
        filename = row['label'].split('/')[-1]
        new_labels = torch.zeros(subject['label'][tio.DATA].shape)
        for indexes in class_index:
            new_labels[indexes[0] == subject['label'][tio.DATA]] = indexes[1]
        o = tio.LabelMap(tensor=new_labels.numpy(),affine=subject["label"].affine)
        o.save('./test_correct_database/' + filename)
