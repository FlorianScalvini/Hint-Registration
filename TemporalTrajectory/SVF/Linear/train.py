#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import torch
import shutil
import argparse
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from dataset import OneWrappedSubjectDataset
from utils import get_model_from_string, create_directory
from Registration import RegistrationModuleSVF
from temporal_trajectory import TemporalTrajectorySVF


# %% Lightning module
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def train_main(config):
    config = json.load(open(args.config))
    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(config['train']['inshape']),
        tio.OneHot(config['train']['num_classes']),
    ])

    ## Dateset's configuration : Load the dataset and the dataloader
    # Load the dataset that return a subject with dataset size = 1
    dataset = OneWrappedSubjectDataset(dataset_path=config['train']['csv_path'], transform=train_transform, lambda_age=lambda x: (x - config['train']['t0']) / (config['train']['t1'] - config['train']['t0']))
    loader = tio.SubjectsLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)
    in_shape = dataset.dataset[0]['image'][tio.DATA].shape[1:] # Get the shape of the image

    try:
        regnet = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)

        if "load" in config['train'] and config['train']['load'] != "":
            state_dict = torch.load(config['train']['load'])
            regnet.load_state_dict(state_dict)

    except:
        raise ValueError("Model initialization failed")


    ## Configuration of the training with hyperparameters
    trainer_args = {
        'max_epochs': config['train']['epochs'],
        'precision': config['train']['precision'],
        'accumulate_grad_batches': config['train']['accumulate_grad_batches'],
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./" + config['train']['logger'], name=None),
        'check_val_every_n_epoch': 20
    }
    trainer_reg = pl.Trainer(**trainer_args)

    # Create the save path : Model parameter and config files
    save_path = trainer_args['logger'].log_dir.replace(config['train']['logger'], "Results")
    create_directory(save_path)
    shutil.copy(args.config, save_path + "/config.json")


    # Train the model
    training_module = TemporalTrajectorySVF(
        regnet=regnet,
        loss=config['train']['loss'],
        lambda_sim=config['train']['lam_l'],
        lambda_seg=config['train']['lam_s'],
        lambda_mag=config['train']['lam_m'],
        lambda_grad=config['train']['lam_g'],
        save_path=save_path)

    trainer_reg.fit(training_module, train_dataloaders=loader, val_dataloaders=loader)

    if 'save' in config['train']:
        torch.save(training_module.reg_model.state_dict(), os.path.join(save_path, config['train']['save'] + "_reg.pth"))

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config.json')
    args = parser.parse_args()
    train_main(config=args.config)

