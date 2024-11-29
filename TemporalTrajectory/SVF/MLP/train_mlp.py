#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import monai
import argparse
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from dataset import PairwiseSubjectsDataset, RandomPairwiseSubjectsDataset
from utils import config_dict_to_tensorboard
from dataset import TripletSubjectDataset, RandomTripletSubjectDataset, TripletStaticAnchorsDataset, subjects_from_csv, WrappedSubjectDataset
from utils import get_cuda_is_available_or_cpu, create_directory, get_model_from_string, get_activation_from_string
from Registration import RegistrationModuleSVF
from temporal_trajectory_mlp import TemporalTrajectoryMLPSVF
from network import MLP
import json
import uuid
# %% Lightning module
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"





def train_main(config):
    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128)
    ])

    config_train = config['train']


    ## Config model
    # get the spatial dimension of the data (3D)
    dataset = WrappedSubjectDataset(dataset_path=config_train['csv_path'], transform=train_transform, lambda_age=lambda x: (x - config_train['t0']) / (config_train['t1'] - config_train['t0']))
    loader = tio.SubjectsLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)

    ## Config model
    # get the spatial dimension of the data (3D)
    in_shape = dataset.dataset[0]['image'][tio.DATA].shape[1:]

    try:
        mlp_model = MLP(hidden_size=config['model_svf']['args']['hidden_size'],
                        activation_layer=get_activation_from_string(config['model_svf']['args']['activation_layer']),
                        output_activation=get_activation_from_string(config['model_svf']['args']['output_activation']))

        reg_net = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)
        state_dict = torch.load(config_train['load'])
        reg_net.load_state_dict(state_dict)

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

    model = TemporalTrajectoryMLPSVF(
        reg_model=reg_net,
        mlp_model=mlp_model,
        loss=config_train['loss'],
        lambda_sim=config_train['lam_l'],
        lambda_seg=config_train['lam_s'],
        lambda_mag=config_train['lam_m'],
        lambda_grad=config_train['lam_g'],
        save_path=save_path)


    trainer_reg = pl.Trainer(**trainer_args)


    config_dict_to_tensorboard(config['model_reg'], trainer_reg.logger.experiment, "Registration model config")
    config_dict_to_tensorboard(config['model_svf'], trainer_reg.logger.experiment, "MLP model config")
    config_dict_to_tensorboard(config_train, trainer_reg.logger.experiment, "Training config")

    # %%
    trainer_reg.fit(model, loader, val_dataloaders=None)
    if config_train['save']:
        torch.save(model.reg_model.state_dict(), os.path.join(save_path, config_train['save'] + "_reg.pth"))
        torch.save(model.temporal_mlp.state_dict(), os.path.join(save_path, config_train['save'] + "_mlp.pth"))



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train_main(config=config)


    # o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    # o.save('source.nii.gz')
    # o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    # o.save('target.nii.gz')
