#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import json
import torch
import monai
import argparse
import numpy as np
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from dataset import subjects_from_csv
from Registration import RegistrationModuleSVF, SpatialTransformer
import torchvision.transforms.functional as TF
from utils import normalize_to_0_1, get_cuda_is_available_or_cpu, create_directory, config_dict_to_markdown, write_text_to_file, get_model_from_string,  map_labels_to_colors, seg_map_error

def warp(tensor: torch.Tensor, flow: torch.Tensor, mode='bilinear') -> torch.Tensor:
    spatial_transformer = SpatialTransformer(size=(128,128,128)).cuda()
    return spatial_transformer(tensor, flow, mode=mode)

def test(config):
    # Config Dataset / Dataloader
    config_test = config['test']
    device = get_cuda_is_available_or_cpu()
    loggers = pl.loggers.TensorBoardLogger(save_dir= "./" + config_test['logger'], name=None)
    save_path = loggers.log_dir.replace(config_test['logger'], "Results")
    create_directory(save_path)

    transforms = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20)
    ]

    text_md = config_dict_to_markdown(config['test'], "Test config")
    loggers.experiment.add_text(text_md)
    text_md = config_dict_to_markdown(config['model_reg'], "Registration model config")
    loggers.experiment.add_text(text_md)
    write_text_to_file(text_md, os.path.join(save_path, "config.md"), mode='w')

    subjects = subjects_from_csv(config_test['csv_path'], age=True, lambda_age=lambda x: (x - config_test['t0']) / (config_test['t1'] - config_test['t0']))
    subjects_set = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms=transforms))
    subjects_set_orig = tio.SubjectsDataset(subjects, transform=None)
    in_shape = subjects_set[0]['image'][tio.DATA].shape[1:]
    source = subjects_set[0]
    target = subjects_set[-1]
    source_image = torch.unsqueeze(source["image"][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(source["label"][tio.DATA], 0).to(device)
    target_image = torch.unsqueeze(target["image"][tio.DATA], 0).to(device).float()

    # Load models
    try:
        model = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)
        state_dict = torch.load(config_test['load'])
        model.load_state_dict(state_dict)

    except:
        raise ValueError("Model initialization failed")
    model.eval().to(device)

    dice_score = DiceMetric(include_background=False, reduction="none")
    with open(save_path + "/results.csv", mode='w') as file:
        header = ["time", "mDice", "Cortex", "Ventricule", "all"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        with torch.no_grad():
            velocity = model(source_image, target_image)
            for i in range(len(subjects_set)):
                other_target = subjects_set[i]
                age = int(other_target['age'] * (config_test['t1'] - config_test['t0']) + config_test['t0'])
                weighted_velocity = velocity * other_target['age']
                forward_flow, backward_flow = model.velocity_to_flow(velocity=weighted_velocity)
                warped_source_label = model.warp(source_label.float(), forward_flow)
                warped_source_image = model.warp(source_image.float(), forward_flow)
                '''
                new_subject = tio.Subject(
                    image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source['image'].affine),
                    label=tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu(), affine=source['label'].affine)
                )
                new_subject = reverse_transforms(new_subject)

                label_map = tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu().numpy(),
                                         affine=source['label'].affine)
                label_map.save(save_path + "/" + str(i) + "_fused_source_label.nii.gz")
                image_wrap = new_subject['image']
                image_wrap.save(save_path + "/" + str(i) + "_fused_source_image.nii.gz")

                new_subject = transforms_onehot(new_subject)
                target_inter = transforms_onehot(subjects_set_orig[i])

                colored_error_seg = map_labels_to_colors(
                    seg_map_error(new_subject["label"][tio.DATA].unsqueeze(0), target_inter["label"][tio.DATA].unsqueeze(0),
                                  dim=1)).squeeze().permute(3, 0, 1, 2)

                loggers.experiment.add_image("Atlas Sagittal Plane",
                                             TF.rotate(colored_error_seg[:, int(in_shape[0] / 2), :, :], 90), age)
                loggers.experiment.add_image("Atlas Coronal Plane",
                                             TF.rotate(colored_error_seg[:, :, int(in_shape[1] / 2), :], 90), age)
                loggers.experiment.add_image("Atlas Axial Plane",
                                             TF.rotate(colored_error_seg[:, :, :, int(in_shape[2] / 2)], 90), age)

                dice = dice_score(new_subject["label"][tio.DATA].unsqueeze(0).to(device),
                                  target_inter["label"][tio.DATA].unsqueeze(0).float().to(device))
                '''
                dice = dice_score(torch.round(warped_source_label).to(device),
                                  other_target["label"][tio.DATA].unsqueeze(0).float().to(device))
                writer.writerow({
                    "time": age,
                    "mDice": torch.mean(dice[0]).item(),
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
    parser = argparse.ArgumentParser(description='Registration Test 3D Longitudinal Images ')
    parser.add_argument('--config', type=str, help='Path to the config file', default='/home/florian/Documents/Programs/Hint-Registration/TemporalTrajectory/SVF/Linear/config_test.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    test(config=config)

