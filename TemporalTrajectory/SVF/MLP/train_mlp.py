#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import monai
import shutil
import argparse
import torchio as tio
from network import MLP
import pytorch_lightning as pl
from temporal_trajectory_mlp import TemporalTrajectoryMLPSVF, TemporalTrajectoryActiveLearningMLP
from dataset import TripletSubjectDataset, RandomTripletSubjectDataset, TripletStaticAnchorsDataset, subjects_from_csv, WrappedSubjectDataset, OneWrappedSubjectDataset
from utils import get_cuda_is_available_or_cpu, create_directory, get_model_from_string, get_activation_from_string, write_text_to_file, config_dict_to_markdown
from Registration import RegistrationModuleSVF

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def train_main(config):
    config = json.load(open(args.config))

    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(config['train']['inshape']),
        tio.OneHot(config['train']['num_classes'])
    ])

    ## Config model
    # get the spatial dimension of the data (3D)
    dataset = OneWrappedSubjectDataset(dataset_path=config['train']['csv_path'], transform=train_transform, lambda_age=lambda x: (x - config['train']['t0']) / (config['train']['t1'] - config_train['t0']))
    loader = tio.SubjectsLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)
    in_shape = dataset.dataset[0]['image'][tio.DATA].shape[1:]
    try:
        mlp_model = MLP(hidden_size=config['model_svf']['args']['hidden_size'],
                        activation_layer=get_activation_from_string(config['model_svf']['args']['activation_layer']),
                        output_activation=get_activation_from_string(config['model_svf']['args']['output_activation']))
        if "load_mlp" in config['train'] and config['train']['load_mlp'] != "":
            state_dict = torch.load(config['train']['load_mlp'])
            mlp_model.load_state_dict(state_dict)
        regnet = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)
        if "load" in config['train'] and config['train']['load'] != "":
            state_dict = torch.load(config['train']['load'])
            regnet.load_state_dict(state_dict)
    except:
        raise ValueError("Model initialization failed")


    ## Config training
    # %%
    trainer_args = {
        'max_epochs': config['train']['epochs'],
        'precision': config['train']['precision'],
        'accumulate_grad_batches': config['train']['accumulate_grad_batches'],
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./" + config['train']['logger'], name=None),
        'check_val_every_n_epoch': 20
    }

    save_path = trainer_args['logger'].log_dir.replace(config['train']['logger'], "Results")
    create_directory(save_path)


    trainer_reg = pl.Trainer(**trainer_args)
    shutil.copy(args.config, save_path + "/config.json")

    # Train the model
    training_module = TemporalTrajectoryMLPSVF(
        regnet=regnet,
        mlp_model=mlp_model,
        loss=config['train']['loss'],
        lambda_sim=config['train']['lam_l'],
        lambda_seg=config['train']['lam_s'],
        lambda_mag=config['train']['lam_m'],
        lambda_grad=config['train']['lam_g'],
        save_path=save_path,
        num_classes=config['train']['num_classes'],
        num_inter_by_epoch=config['train']['num_inter_by_epoch']
    )
    trainer_reg.fit(training_module, loader, val_dataloaders=loader)

    if 'save' in config['train']:
        torch.save(training_module.reg_model.state_dict(), os.path.join(save_path, config['train']['save'] + "_reg.pth"))
        torch.save(training_module.temporal_mlp.state_dict(), os.path.join(save_path, config['train']['save'] + "_mlp.pth"))



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config.json')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    train_main(config=args.config)
