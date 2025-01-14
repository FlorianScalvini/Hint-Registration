#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import csv
import json
import monai
import torch
import argparse
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from dataset import subjects_from_csv
from temporal_trajectory_mlp import MLP
import torchvision.transforms.functional as TF
from Registration import RegistrationModuleSVF
from utils import get_cuda_is_available_or_cpu, create_directory, seg_map_error, map_labels_to_colors, write_namespace_arguments

def test(args):
    device = get_cuda_is_available_or_cpu()
    loggers = pl.loggers.TensorBoardLogger(save_dir= "./log" , name=None)
    save_path = loggers.log_dir.replace('log', "Results")
    create_directory(save_path)

    ## Config Dataset / Dataloader
    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(args.inshape),
        tio.OneHot(args.num_classes)
    ])

    subjects_list = subjects_from_csv(dataset_path=args.csv, lambda_age=lambda x: (x -args.t0) / (args.t1 - args.t0))
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    reverse_transform = tio.Compose([
        tio.Resize(221),
        tio.CropOrPad(target_shape=subjects_dataset[0]['image'][tio.DATA].shape[1:]),
        tio.OneHot(args.num_classes)
    ])

    in_shape = subjects_dataset[0]['image'][tio.DATA].shape[1:]
    source_subject = None
    target_subject = None
    for s in subjects_dataset:
        if s.age == 0:
            source_subject = s
        if s.age == 1:
            target_subject = s

    # Load models
    reg_model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32], strides=[2,2]), inshape=in_shape, int_steps=7)
    reg_model.load_state_dict(torch.load(args.load))
    reg_model.eval().to(device)

    mlp_model = None
    if args.mode == 'mlp':
        mlp_model = MLP(hidden_size=args.mlp_hidden_size)
        mlp_model.load_state_dict(torch.load(args.load_mlp))
        mlp_model.eval().to(device)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    with open(save_path + "/results.csv", mode='w') as file:
        header = ["time", "mDice", "Cortex", "Ventricule", "all"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        with torch.no_grad():
            source_subject = transforms(source_subject)
            target_subject = transforms(target_subject)
            source_image = torch.unsqueeze(source_subject["image"][tio.DATA], 0).to(device)
            source_label = torch.unsqueeze(source_subject["label"][tio.DATA], 0).to(device)
            velocity = reg_model(source_image, torch.unsqueeze(target_subject["image"][tio.DATA], 0).to(device).float())
            for target_subject in subjects_dataset:
                age = int(target_subject['age'] * (args.t1 - args.t0) + args.t0)
                transformed_target_subject = transforms(target_subject)
                if args.mode == 'mlp':
                    weighted_age = mlp_model(torch.tensor([transformed_target_subject['age']]).to(device))
                else:
                    weighted_age = transformed_target_subject['age']
                forward_flow, backward_flow = reg_model.velocity_to_flow(velocity=velocity * weighted_age)
                warped_source_label = reg_model.warp(source_label.float(), forward_flow)
                warped_source_image = reg_model.warp(source_image.float(), forward_flow)
                warped_subject = tio.Subject(
                    image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source_subject['image'].affine),
                    label=tio.LabelMap(tensor=torch.argmax(torch.round(warped_source_label), dim=1).int().detach().cpu(), affine=source_subject['label'].affine)
                )
                warped_subject = reverse_transform(warped_subject)
                if args.save_image:
                    warped_subject['image'].save(save_path + "/" + str(age) + "_fused_source_image.nii.gz")
                    warped_subject['label'].save(save_path + "/" + str(age) + "_fused_source_label.nii.gz")
                if args.error_map:
                    colored_error_seg = map_labels_to_colors(
                        seg_map_error(warped_subject["label"][tio.DATA].unsqueeze(0), warped_subject["label"][tio.DATA].unsqueeze(0),
                                      dim=1)).squeeze().permute(3, 0, 1, 2)

                    loggers.experiment.add_image("Atlas Sagittal Plane",
                                                 TF.rotate(colored_error_seg[:, int(in_shape[0] / 2), :, :], 90), age)
                    loggers.experiment.add_image("Atlas Coronal Plane",
                                                 TF.rotate(colored_error_seg[:, :, int(in_shape[1] / 2), :], 90), age)
                    loggers.experiment.add_image("Atlas Axial Plane",
                                                 TF.rotate(colored_error_seg[:, :, :, int(in_shape[2] / 2)], 90), age)

                dice = dice_metric(torch.round(warped_source_label).to(device), target_subject["label"][tio.DATA].unsqueeze(0).float().to(device))
                writer.writerow({
                    "time": age,
                    "mDice": torch.mean(dice[0][1:]).item(),
                    "Cortex": torch.mean(dice[0][3:5]).item(),
                    "Ventricule": torch.mean(dice[0][7:9]).item(),
                    "all": dice[0].cpu().numpy()
                })
                loggers.experiment.add_scalar("Dice ventricule", torch.mean(dice[0][7:9]).item(), age)
                loggers.experiment.add_scalar("Dice cortex", torch.mean(dice[0][3:5]).item(), age)
                loggers.experiment.add_scalar("mDice", torch.mean(dice[0]).item(), age)






# %% Main program
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='../../../data/full_dataset.csv')
    parser.add_argument('--t0', type=int, help='Initial time point', default=21)
    parser.add_argument('--t1', type=int, help='Final time point', default=36)
    parser.add_argument('--load', type=str, help='Path to the model', default='')
    parser.add_argument('--load_mlp', type=str, help='Path to the mlp model', default='')
    parser.add_argument('--save', type=str, help='Name of the model', default='final')
    parser.add_argument('--inshape', type=int, help='Size of the input image', default=128)
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--error_map', type=bool, help='Compute the error map', default=False)
    parser.add_argument('--mode', type=str, help='SVF Temporal mode', choices={'mlp', 'linear'}, default='mlp')
    parser.add_argument('--save_image', type=bool, help='Save MRI', default=False)
    parser.add_argument('--mlp_hidden_size', type=int, nargs='+', help='Hidden size of the MLP model', default=[1, 32, 32, 32, 1])
    args = parser.parse_args()
    test(args=args)

