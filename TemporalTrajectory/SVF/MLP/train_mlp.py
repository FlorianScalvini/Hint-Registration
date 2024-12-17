#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import monai
import argparse
import torchio as tio
from network import MLP
import pytorch_lightning as pl
from temporal_trajectory_mlp import TemporalTrajectoryMLPSVF, TemporalTrajectoryActiveLearningMLP
from dataset import TripletSubjectDataset, RandomTripletSubjectDataset, TripletStaticAnchorsDataset, subjects_from_csv, WrappedSubjectDataset
from utils import get_cuda_is_available_or_cpu, create_directory, get_model_from_string, get_activation_from_string, write_text_to_file, config_dict_to_markdown
from Registration import RegistrationModuleSVF

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def train_main(config):
    ## Config Dataset / Dataloader
    config_train = config['train']
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(config_train['num_classes'])
    ])

    ## Config model
    # get the spatial dimension of the data (3D)
    dataset = WrappedSubjectDataset(dataset_path=config_train['csv_path'], transform=train_transform, lambda_age=lambda x: (x - config_train['t0']) / (config_train['t1'] - config_train['t0']))
    loader = tio.SubjectsLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)
    in_shape = dataset.dataset[0]['image'][tio.DATA].shape[1:]
    try:
        mlp_model = MLP(hidden_size=config['model_svf']['args']['hidden_size'],
                        activation_layer=get_activation_from_string(config['model_svf']['args']['activation_layer']),
                        output_activation=get_activation_from_string(config['model_svf']['args']['output_activation']))
        if "load_mlp" in config_train and config_train['load_mlp'] != "":
            state_dict = torch.load(config_train['load_mlp'])
            mlp_model.load_state_dict(state_dict)
        regnet = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)
        if "load" in config_train and config_train['load'] != "":
            state_dict = torch.load(config_train['load'])
            regnet.load_state_dict(state_dict)
    except:
        raise ValueError("Model initialization failed")


    ## Config training
    # %%
    trainer_args = {
        'max_epochs': config_train['epochs'],
        'precision': config_train['precision'],
        'accumulate_grad_batches': config_train['accumulate_grad_batches'],
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./" + config_train['logger'], name=None)
    }

    save_path = trainer_args['logger'].log_dir.replace(config_train['logger'], "Results")
    create_directory(save_path)


    trainer_reg = pl.Trainer(**trainer_args)

    text_md = config_dict_to_markdown(config['model_reg'], "Registration model config")
    trainer_reg.logger.experiment.add_text(text_md)
    write_text_to_file(text_md, os.path.join(save_path, "config.md"), mode='w')
    text_md = config_dict_to_markdown(config['model_svf'], "MLP model config")
    trainer_reg.logger.experiment.add_text(text_md)
    write_text_to_file(text_md, os.path.join(save_path, "config.md"), mode='a')
    text_md = config_dict_to_markdown(config_train, "Training config")
    trainer_reg.logger.experiment.add_text(text_md)
    write_text_to_file(text_md, os.path.join(save_path, "config.md"), mode='a')

    # Train the model
    training_module = TemporalTrajectoryMLPSVF(
        regnet=regnet,
        mlp_model=mlp_model,
        loss=config_train['loss'],
        lambda_sim=config_train['lam_l'],
        lambda_seg=config_train['lam_s'],
        lambda_mag=config_train['lam_m'],
        lambda_grad=config_train['lam_g'],
        save_path=save_path,
        num_classes=config_train['num_classes'],
        num_inter_by_epoch=config_train['num_inter_by_epoch']
    )
    trainer_reg.fit(training_module, loader, val_dataloaders=None)

    if 'save' in config_train:
        torch.save(training_module.reg_model.state_dict(), os.path.join(save_path, config_train['save'] + "_reg.pth"))
        torch.save(training_module.temporal_mlp.state_dict(), os.path.join(save_path, config_train['save'] + "_mlp.pth"))



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    torch.set_float32_matmul_precision('high')
    train_main(config=config)
