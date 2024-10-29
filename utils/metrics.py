import torch
from torch import Tensor
from torchmetrics import F1Score
import logging
import torchio as tio
import pandas as pd
import argparse

class SegmentationMetric():
    def __init__(self, labels : dict[int,str], list_class : list[int]):
        self.metric = F1Score(task="multiclass", num_classes=len(labels), average=None)
        self.list_class = list_class
        self.labels = labels

    def update(self, preds: Tensor, targets: Tensor) -> None:
        self.metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        data = self.metric.compute()
        metric = {labels[idx]: data[idx].numpy().item() for idx in list_class}
        metric['average'] = torch.mean(data).numpy().item()
        return metric

    def __str__(self):
        result = [f"{k} : {v}" for k, v in self.compute().items()]
        # Join the list into a single string with newline characters
        return "\n".join(result)

    def reset(self) -> None:
        self.metric.reset()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = False,
                        default="/media/ubuntu/SSD Samsung/Postdoc/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t23.00_dhcp-19.nii.gz")
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = False,
                        default="/home/ubuntu/Documents/Result/Biblio/ANTS/ants_warped_target_label.nii.gz")
    parser.add_argument( '--list_class', help='Number of classes', type=str, required = False,
                        default="3,4,5,6")
    parser.add_argument( '--tsv_labelname', help='List of label names', type=str, required = False,
                        default="/media/ubuntu/SSD Samsung/Postdoc/Dataset/template_dHCP/fetal_brain_mri_atlas/info/dhcp-atlas-summary-info-19-labels.csv")

    args = parser.parse_args()


    subject_target = tio.Subject(
        label=tio.LabelMap(args.target),
    )

    subject_source = tio.Subject(
        label=tio.LabelMap(args.source),
    )

    transform = tio.Compose(
        [
            tio.CropOrPad(192),
            tio.Resize(175)
        ]
    )

    #subject_target = transform(subject_target)


    df = pd.read_csv(args.tsv_labelname, sep=',')
    labels = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    list_class = []
    try:
        list_class = list(map(int, args.list_class.split(',')))
    except ValueError:
        logging.exception("An exception was thrown!")
    metric = SegmentationMetric(labels=labels, list_class=list_class)
    metric.update(subject_source["label"][tio.DATA], subject_target["label"][tio.DATA])
    print(metric)
