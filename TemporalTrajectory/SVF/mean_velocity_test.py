#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import monai
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import torchvision.transforms.functional as TF

from dataset import subjects_from_csv
from utils import dice_score, normalize_to_0_1
from Registration import RegistrationModuleSVF
from utils import Grad3d, get_cuda_is_available_or_cpu
from temporal_trajectory import TemporalTrajectorySVF

NUM_CLASSES = 20

def test(arguments):
    loggers = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=arguments.logger)

    transforms = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20),
    ]

    subjects_set = tio.SubjectsDataset(subjects=subjects_from_csv(dataset_path=arguments.csv_path,
                                                                  age=True,
                                                                  lambda_age=lambda x: (x - arguments.t0) / (arguments.t1 - arguments.t0)),
                                       transform=tio.Compose(transforms=transforms))

    velocity_calc_subjects_set = tio.SubjectsDataset(subjects=subjects_from_csv(dataset_path=arguments.csv_path_velocity_calc,
                                                                  age=True,
                                                                  lambda_age=lambda x: (x - arguments.t0) / (arguments.t1 - arguments.t0)),
                                       transform=tio.Compose(transforms=transforms))

    in_shape = subjects_set[0]['image'][tio.DATA].shape[1:]
    model = RegistrationModuleSVF(model=monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=in_shape, int_steps=7).to(get_cuda_is_available_or_cpu())
    try:
        model.load_state_dict(torch.load(arguments.load))
    except:
        raise ValueError("No model to load or model not compatible")


    # calculate mean velocity:
    denum = 0
    num = torch.zeros([1, 3] + list(in_shape)).to(model.device)
    weightedMeanVelocity = None
    transformationPairs = list(itertools.combinations(range(len(velocity_calc_subjects_set)), 2))


    with torch.no_grad():
        for i, j in transformationPairs:
            sample_i = velocity_calc_subjects_set[i]
            sample_j = velocity_calc_subjects_set[j]
            time_ij = sample_j['age']- sample_i['age']
            velocity_ij = model(sample_i['image'][tio.DATA].float().unsqueeze(dim=0).to(model.device), sample_j['image'][tio.DATA].unsqueeze(dim=0).float().to(model.device))
            num += velocity_ij * time_ij
            denum += time_ij * time_ij
        weightedMeanVelocity = num / denum if denum != 0 else torch.zeros_like(num)


    source = None

    for s in subjects_set:
        if s.age == 0:
            source = s
            break

    source_image = torch.unsqueeze(source["image"][tio.DATA], 0).to(model.device)
    source_label = torch.unsqueeze(source["label"][tio.DATA], 0).to(model.device)


    for i in range(len(subjects_set)):
        other_target = subjects_set[i]
        diff_age = other_target['age'] - source['age']

        forward_flow, backward_flow = model.velocity_to_flow(velocity=weightedMeanVelocity * diff_age)
        wrapped_source = model.wrap(source_image, forward_flow)
        wrapped_source_label = model.wrap(source_label.float(), forward_flow)

        dice = dice_score(torch.argmax(wrapped_source_label, dim=1), torch.argmax(other_target['label'][tio.DATA].to(model.device), dim=0), num_classes=NUM_CLASSES)
        source_inter_warped_image = normalize_to_0_1(other_target['image'][tio.DATA])
        img = normalize_to_0_1(wrapped_source[0].detach())

        real_age = int(other_target['age'] * (arguments.t1 - arguments.t0) + arguments.t0)
        loggers.experiment.add_image("Target Sagittal Plane", TF.rotate(source_inter_warped_image[:, int(in_shape[0] / 2), :, :], 90), real_age)
        loggers.experiment.add_image("Target Coronal Plane", TF.rotate(source_inter_warped_image[:, :, int(in_shape[1] / 2), :], 90),real_age)
        loggers.experiment.add_image("Target Axial Plane", TF.rotate(source_inter_warped_image[:, :, :, int(in_shape[2] / 2)], 90), real_age)

        loggers.experiment.add_image("Wrapped Sagittal Plane", TF.rotate(img[:, int(in_shape[0] / 2), :, :], 90), real_age)
        loggers.experiment.add_image("Wrapped Coronal Plane", TF.rotate(img[:, :, int(in_shape[1] / 2), :], 90), real_age)
        loggers.experiment.add_image("Wrapped Axial Plane", TF.rotate(img[:, :, :, int(in_shape[2] / 2)], 90), real_age)

        loggers.experiment.add_scalar("Dice white matter", np.mean(dice[5:7]), real_age)
        loggers.experiment.add_scalar("Dice cortex", np.mean(dice[3:5]), real_age)
        loggers.experiment.add_scalar("mDice", np.mean(dice), real_age)

        print(real_age)


# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Registration 3D Longitudinal Images : Inference')
    parser.add_argument('-i', '--csv_path', help='csv file ', type=str, required=False,
                        default="./train_dataset_long.csv")
    parser.add_argument('-p', '--csv_path_velocity_calc', help='csv file ', type=str, required=False,
                        default="./train_dataset_3_inter.csv")
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required=False, default=24)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required=False, default=32)

    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./save/")
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./model.pth")
    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")


    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')

    test(arguments=args)
