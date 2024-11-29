#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import torch
import monai
import argparse
import numpy as np
import torchio as tio
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
import json
from dataset import subjects_from_csv
from Registration import RegistrationModuleSVF
from temporal_trajectory_mlp import MLP
from utils import dice_score, dice_score_old, normalize_to_0_1, get_cuda_is_available_or_cpu, get_activation_from_string, get_model_from_string, create_directory, seg_map_error, map_labels_to_colors, config_dict_to_tensorboard

NUM_CLASSES = 3

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
        tio.OneHot(3),
    ]

    config_dict_to_tensorboard(config['test'], loggers.experiment, "Test config")
    config_dict_to_tensorboard(config['model_reg'], loggers.experiment, "Registration model config")
    config_dict_to_tensorboard(config['model_svf'], loggers.experiment, "MLP model config")

    subjects = subjects_from_csv(config_test['csv_path'], age=True, lambda_age=lambda x: (x - config_test['t0']) / (config_test['t1'] - config_test['t0']))
    subjects_set = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms=transforms))
    in_shape = subjects_set[0]['image'][tio.DATA].shape[1:]
    source = subjects_set[0]
    target = subjects_set[-1]
    source_image = torch.unsqueeze(source["image"][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(source["label"][tio.DATA], 0).to(device)
    target_image = torch.unsqueeze(target["image"][tio.DATA], 0).to(device).float()
    target_label = torch.unsqueeze(target["label"][tio.DATA], 0).to(device).float()

    # Load models
    try:
        mlp_model = MLP(hidden_size=config['model_svf']['args']['hidden_size'],
                        activation_layer=get_activation_from_string(config['model_svf']['args']['activation_layer']),
                        output_activation=get_activation_from_string(config['model_svf']['args']['output_activation']))
        mlp_model.load_state_dict(torch.load(config_test['load_mlp']))
        reg_net = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)
        state_dict = torch.load(config_test['load'])
        reg_net.load_state_dict(state_dict)

    except:
        raise ValueError("Model initialization failed")
    mlp_model.eval().to(device)
    reg_net.eval().to(device)


    # Calculate velocity
    with torch.no_grad():
        velocity = reg_net(source_image, target_image)


    for i in range(len(subjects_set)):
        other_target = subjects_set[i]
        coef = mlp_model(torch.asarray([other_target['age']]).to(device))
        weighted_velocity = velocity * coef

        forward_flow, backward_flow = reg_net.velocity_to_flow(velocity=weighted_velocity)
        warped_source_label = reg_net.warp(source_label.float(), forward_flow, mode='nearest')
        colored_error_seg = map_labels_to_colors(seg_map_error(warped_source_label, other_target['label'][tio.DATA].unsqueeze(dim=0).to(device))).squeeze(dim=0).permute(3,0,1,2)
        real_age = int(other_target['age'] * (config_test['t1'] - config_test['t0']) + config_test['t0'])


        loggers.experiment.add_image("Atlas Sagittal Plane", TF.rotate(colored_error_seg[:, int(in_shape[0] / 2), :, :], 90), real_age)
        loggers.experiment.add_image("Atlas Coronal Plane", TF.rotate(colored_error_seg[:, :, int(in_shape[1] / 2), :], 90), real_age)
        loggers.experiment.add_image("Atlas Axial Plane", TF.rotate(colored_error_seg[:, :, :, int(in_shape[2] / 2)], 90), real_age)

        old = dice_score_old(torch.argmax(warped_source_label.squeeze(dim=0),dim=0), torch.argmax(other_target['label'][tio.DATA].cuda(), dim=0), num_classes=3)
        new = dice_score(warped_source_label.squeeze(dim=0).detach().cpu().numpy(), other_target['label'][tio.DATA].detach().cpu().numpy())
        print(real_age, old, new)


        """
        
        warped_source_image = reg_net.warp(source_image.float(), forward_flow)
        o = tio.ScalarImage(tensor=warped_source_image.detach().numpy(), affine=subjects_set[0]['source'].affine)
        o.save(os.path.join(save_path, str(real_age) + "_image.noo.gz"))
        """
        print(real_age)



# %% Main program
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--config', type=str, help='Path to the config file', default='/home/florian/Documents/Programs/Hint-Registration/TemporalTrajectory/SVF/MLP/config_test.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    test(config=config)

