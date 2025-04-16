#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os.path
import argparse
import monai
import torch
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.nn.functional import dropout

from modules.pairwise_registration import PairwiseRegistrationModuleVelocity
from modules.longitudinal_deformation import OurLongitudinalDeformation
from utils import get_cuda_is_available_or_cpu, create_directory, write_namespace_arguments, subjects_from_csv


def test(args):
    device = get_cuda_is_available_or_cpu()
    loggers = pl.loggers.TensorBoardLogger(save_dir= "./log" , name=None)
    save_path = loggers.log_dir.replace('log', "Results")
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "configs.json"))
    if args.save_image:
        create_directory(os.path.join(save_path, "images"))
        create_directory(os.path.join(save_path, "flow"))
        create_directory(os.path.join(save_path, "label"))
    ## Config Dataset / Dataloader

    subjects_list = subjects_from_csv(dataset_path=args.csv, lambda_age=lambda x: (x -args.t0) / (args.t1 - args.t0))
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    transforms = tio.Compose([
        tio.CropOrPad(target_shape=args.csize),
        tio.Resize(target_shape=args.rsize),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.OneHot(args.num_classes)
    ])

    reverse_transform = tio.Compose([
        tio.Resize(target_shape=args.csize),
        tio.CropOrPad(target_shape=subjects_dataset[0]["image"][tio.DATA].shape[1:]),
        tio.OneHot(args.num_classes)
    ])

    source_subject = None
    target_subject = None
    for s in subjects_dataset:
        if s.age == 0:
            source_subject = s
        if s.age == 1:
            target_subject = s

    # Load models
    model = OurLongitudinalDeformation(
        time_mode=args.time_mode, hidden_dim=args.mlp_hidden_dim, t0=args.t0, t1=args.t1,
        reg_model=PairwiseRegistrationModuleVelocity(model=monai.networks.nets.AttentionUnet(dropout=0.1,
            spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32, 64], strides=[2, 2, 2]), int_steps=7))
    model.eval().to(device)
    model.reg_model.load_state_dict(torch.load(args.load))
    if args.time_mode == 'mlp':
        model.load_temporal(args.load_mlp)
    dice_metric = DiceMetric(include_background=True, reduction="none")
    with open(save_path + "/results.csv", mode='w') as file:
        header = ["time", "mDice", "Cortex", "Ventricule", "all"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        with torch.no_grad():
            source_subject = transforms(source_subject)
            target_subject_transformed = transforms(target_subject)
            source_image = torch.unsqueeze(source_subject["image"][tio.DATA], 0).to(device)
            source_label = torch.unsqueeze(source_subject["label"][tio.DATA], 0).to(device)
            velocity = model.forward((source_image, torch.unsqueeze(target_subject_transformed["image"][tio.DATA], 0).to(device).float()))

            for subject in subjects_dataset:
                age = subject['age'] * (args.t1 - args.t0) + args.t0
                timed_velocity = model.encode_time(torch.Tensor([subject['age']]).to(device)) * velocity
                forward_flow = model.reg_model.velocity2displacement(timed_velocity)
                warped_source_image = model.reg_model.warp(source_image.float(), forward_flow)
                warped_source_label = model.reg_model.warp(source_label.float(), forward_flow)
                warped_subject = tio.Subject(
                    image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source_subject['image'].affine),
                    label=tio.LabelMap(tensor=torch.argmax(torch.round(warped_source_label), dim=1).int().detach().cpu(), affine=source_subject['label'].affine),
                    flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(), affine=source_subject['image'].affine)
                )
                warped_subject = reverse_transform(warped_subject)
                dice = dice_metric(warped_subject['label'][tio.DATA].long().cpu().unsqueeze(0), torch.nn.functional.one_hot(subject['label'][tio.DATA].to(device).long().cpu()).permute(0,4,1,2,3))
                writer.writerow({
                    'time': age,
                    "mDice": torch.mean(dice[0][1:]).item(),
                    "Cortex": torch.mean(dice[0][3:5]).item(),
                    "Ventricule": torch.mean(dice[0][7:9]).item(),
                    "all": dice[0].cpu().numpy()
                })
                print(age, torch.mean(dice[0][1:]).item())
                loggers.experiment.add_scalar("Dice ventricule", torch.mean(dice[0][7:9]).item(), age)
                loggers.experiment.add_scalar("Dice cortex", torch.mean(dice[0][3:5]).item(), age)
                loggers.experiment.add_scalar("mDice", torch.mean(dice[0]).item(), age)
                warped_subject['label'][tio.DATA] = torch.argmax(warped_subject['label'][tio.DATA], dim=0).int().unsqueeze(0)
                if args.save_image:
                    warped_subject['image'].save(os.path.join(save_path, "images", str(age) + "_warped_source_image.nii.gz"))
                    warped_subject['label'].save(os.path.join(save_path, "label", str(age) + "_label.nii.gz"))
                    warped_subject['flow'].save(os.path.join(save_path, "flow", str(age) + "_flow.nii.gz"))

    if model.time_mode == 'mlp':
        x = np.arange(0, 1, 0.01)
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = model.encode_time(torch.Tensor([x[i]]).to(device)).detach().cpu().numpy()
        plt.plot(x, y)
        plt.show()




# %% Main program
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo registration.yaml 3D Longitudinal Images with MLP model')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='/home/florian/Documents/Programs/longitudinal-svf/dataset/dHCP/dataset.csv')
    parser.add_argument('--t0', type=int, help='Initial time point', default=21)
    parser.add_argument('--t1', type=int, help='Final time point', default=36)
    parser.add_argument('--load', type=str, help='Path to the model', default='/home/florian/Documents/Programs/longitudinal-svf/outputs/longitudinal/2025-04-15_22-43-06/last_model_reg.pth')
    parser.add_argument('--load_mlp', type=str, help='Path to the mlp model', default="/home/florian/Documents/Programs/longitudinal-svf/outputs/longitudinal/2025-04-15_22-43-06/last_model_mlp.pth")
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize shape', default=[128, 128, 128])
    parser.add_argument('--csize', type=int, nargs='+', help='Cropsize shape', default=[221, 221, 221])
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--time_mode', type=str, help='SVF Temporal mode', choices={'mlp', 'linear'}, default='mlp')
    parser.add_argument('--save_image', type=bool, help='Save MRI', default=True)
    parser.add_argument('--mlp_hidden_dim', type=int, help='Hidden size of the MLP model', default=[32,32,32]),
    parser.add_argument('--mlp_num_layers', type=int, help='Number layer of the MLP', default=4),
    args = parser.parse_args()
    test(args=args)

