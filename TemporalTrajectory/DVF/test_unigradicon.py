import torch
import argparse
import torchio as tio
import pandas as pd
import torch.nn as nn
from Registration import UniGradIconRegistrationWrapper
import pytorch_lightning as pl
import numpy as np

import torchvision.transforms.functional as TF
from utils import dice_score, normalize_to_0_1
from dataset import subjects_from_csv


def test(arguments):
    loggers = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=arguments.logger)

    transforms = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(175),
        tio.OneHot(20),
    ]

    subjects = subjects_from_csv(arguments.csv_path, age=True, lambda_age=lambda x: (x - arguments.t0) / (arguments.t1 - arguments.t0))
    subjects_set = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms=transforms))

    template_t0 = None
    template_t1 = None
    for s in subjects_set:
        if s.age == 0:
            template_t0 = s
        if s.age == 0.125:
            template_t1 = s


    net = UniGradIconRegistrationWrapper().cuda().eval()

    source_data = torch.unsqueeze(template_t0.image.data, 0).cuda()
    source_data_label = torch.unsqueeze(template_t0.label.data, 0).cuda()
    ref_data = torch.unsqueeze(template_t1.image.data, 0).cuda()
    with torch.no_grad():
        forward_flow, backward_flow = net(source_data, ref_data)

    for i in range(len(subjects_set) - 1):
        target = subjects_set[i + 1]
        warped_source = net.warp(source_data, forward_flow)
        warped_source_label = net.warp(source_data_label.float(), forward_flow)

        inter_age = int(target.age * (32 - 24) + 24)
        img = normalize_to_0_1(warped_source[0].detach())
        loggers.experiment.add_image("Wrapped Sagittal Plane", TF.rotate(img[:, 88, :, :], 90), inter_age)
        loggers.experiment.add_image("Wrapped Coronal Plane", TF.rotate(img[:, :, 88, :], 90), inter_age)
        loggers.experiment.add_image("Wrapped Axial Plane", TF.rotate(img[:, :, :, 88], 90), inter_age)

        dice = dice_score(torch.argmax(warped_source_label, dim=1),
                          torch.argmax(target.label.data.cuda(), dim=0), num_classes=20)
        loggers.experiment.add_scalar("Dice white matter", np.mean(dice[5:7]), inter_age)
        loggers.experiment.add_scalar("Dice cortex", np.mean(dice[3:5]), inter_age)
        loggers.experiment.add_scalar("mDice", np.mean(dice), inter_age)
        print(inter_age, ":", np.mean(dice))
        source_data = warped_source
        source_data_label = warped_source_label



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Registration 3D Longitudinal Images : Inference')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False,
                        default="/home/florian/Documents/Programs/Hint-Registration/data/train_dataset.csv")
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required=False, default=24)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required=False, default=32)
    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./save/")
    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    test(arguments=args)
