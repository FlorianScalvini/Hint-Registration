#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import monai
import argparse
import numpy as np
import torchio as tio
import pytorch_lightning as pl
import torchvision.transforms.functional as TF

from dataset import subjects_from_csv
from Registration import RegistrationModuleSVF

from utils import dice_score, normalize_to_0_1, get_cuda_is_available_or_cpu

NUM_CLASSES = 20

def test(arguments):
    device = 'cpu'
    loggers = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=arguments.logger)

    transforms = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20),
    ]
    subjects = subjects_from_csv(arguments.csv_path, age=True, lambda_age=lambda x: (x - arguments.t0) / (arguments.t1 - arguments.t0))
    subjects_set = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms=transforms))

    in_shape = subjects_set[0]['image'][tio.DATA].shape[1:]
    model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=in_shape, int_steps=7).to(device)
    try:
        model.load_state_dict(torch.load(arguments.load))
    except:
        raise ValueError("No model to load or model not compatible")
    model.eval()
    source = None
    target = None
    for s in subjects_set:
        if s.age == 0:
            source = s
        if s.age == 1:
            target = s

    source_image = torch.unsqueeze(source["image"][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(source["label"][tio.DATA], 0).to(device)

    target_image = torch.unsqueeze(target["image"][tio.DATA], 0).to(device)
    target_label = torch.unsqueeze(target["label"][tio.DATA], 0).to(device)


    velocity = model(source_image, target_image)

    for i in range(len(subjects_set)):
        other_target = subjects_set[i]
        weighted_age = (other_target['age'] - source['age']) / (target['age'] - source['age'])

        forward_flow, backward_flow = model.velocity_to_flow(velocity=velocity * weighted_age)

        warped_source_label = model.warp(source_label.float(), forward_flow)

        dice = dice_score(torch.argmax(warped_source_label, dim=1), torch.argmax(other_target['label'][tio.DATA].to(device), dim=0), num_classes=NUM_CLASSES)
        real_age = int(other_target['age'] * (arguments.t1 - arguments.t0) + arguments.t0)

        '''
        warped_source = model.warp(source_image, forward_flow)
        source_inter_warped_image = normalize_to_0_1(other_target['image'][tio.DATA])
        img = normalize_to_0_1(warped_source[0].detach())
  
        loggers.experiment.add_image("Target Sagittal Plane", TF.rotate(source_inter_warped_image[:, int(in_shape[0] / 2), :, :], 90), real_age)
        loggers.experiment.add_image("Target Coronal Plane", TF.rotate(source_inter_warped_image[:, :, int(in_shape[1] / 2), :], 90),real_age)
        loggers.experiment.add_image("Target Axial Plane", TF.rotate(source_inter_warped_image[:, :, :, int(in_shape[2] / 2)], 90), real_age)

        loggers.experiment.add_image("Warped Sagittal Plane", TF.rotate(img[:, int(in_shape[0] / 2), :, :], 90), real_age)
        loggers.experiment.add_image("Warped Coronal Plane", TF.rotate(img[:, :, int(in_shape[1] / 2), :], 90), real_age)
        loggers.experiment.add_image("Warped Axial Plane", TF.rotate(img[:, :, :, int(in_shape[2] / 2)], 90), real_age)
        '''
        loggers.experiment.add_scalar("Dice white matter", np.mean(dice[5:7]), real_age)
        loggers.experiment.add_scalar("Dice cortex", np.mean(dice[3:5]), real_age)
        loggers.experiment.add_scalar("mDice", np.mean(dice), real_age)

        print(real_age)


# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Registration 3D Longitudinal Images : Inference')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False,
                        default="/home/florian/Documents/Programs/Hint-Registration/data/full_dataset.csv")
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required=False, default=21)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required=False, default=36)

    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./save/")
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./model_linear_best.pth")
    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")

    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')

    test(arguments=args)
