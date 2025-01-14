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
from dataset import OneWrappedSubjectDataset
from Registration import RegistrationModuleSVF
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils import create_directory, write_namespace_arguments
from temporal_trajectory_mlp import TemporalTrajectoryMLPSVF, TemporalTrajectorySVF


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def train_main(args):


    ## Config Dataset / Dataloader
    train_transforms = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(args.inshape),
        tio.OneHot(args.num_classes)
    ])


    ## Config model
    # get the spatial dimension of the data (3D)
    dataset = OneWrappedSubjectDataset(dataset_path=args.csv, transform=train_transforms, lambda_age=lambda x: (x -args.t0) / (args.t1 - args.t0))
    loader = tio.SubjectsLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)
    in_shape = dataset.dataset[0]['image'][tio.DATA].shape[1:]
    try:
        mlp_model = MLP(hidden_size=args.mlp_hidden_size)
        if args.load_mlp != "":
            state_dict = torch.load(args.load_mlp)
            mlp_model.load_state_dict(state_dict)
        reg_model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32], strides=[2,2]), inshape=in_shape, int_steps=7)
        if args.load != "":
            state_dict = torch.load(args.load)
            reg_model.load_state_dict(state_dict)
    except:
        raise ValueError("Model initialization failed")

    ## Config training
    trainer_args = {
        'max_epochs': args.epochs,
        'precision': args.precision,
        'strategy': DDPStrategy(find_unused_parameters=True),
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./log", name=None),
        'check_val_every_n_epoch': 20
    }

    save_path = trainer_args['logger'].log_dir.replace("log", "Results")
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "config.json"))

    if args.mode == 'linear':
        training_module = TemporalTrajectorySVF(
            reg_model=reg_model,
            loss=args.loss,
            lambda_sim=args.lam_l,
            lambda_seg=args.lam_s,
            lambda_mag=args.lam_m,
            lambda_grad=args.lam_g,
            save_path=save_path,
            num_classes=args.num_classes,
            num_inter_by_epoch=args.num_inter_by_epoch
        )
    else:
        training_module = TemporalTrajectoryMLPSVF(
            reg_model=reg_model,
            mlp_model=mlp_model,
            loss=args.loss,
            lambda_sim=args.lam_l,
            lambda_seg=args.lam_s,
            lambda_mag=args.lam_m,
            lambda_grad=args.lam_g,
            save_path=save_path,
            num_classes=args.num_classes,
            num_inter_by_epoch=args.num_inter_by_epoch
        )
    trainer_reg = pl.Trainer(**trainer_args)
    trainer_reg.fit(training_module, loader, val_dataloaders=loader)
    torch.save(training_module.reg_model.state_dict(), os.path.join(save_path, "final_reg.pth"))
    if args.mode == 'mlp':
        torch.save(training_module.temporal_mlp.state_dict(), os.path.join(save_path, "final_mlp.pth"))



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='../../data/full_dataset.csv')
    parser.add_argument('--t0', type=int, help='Initial time point', default=21)
    parser.add_argument('--t1', type=int, help='Final time point', default=36)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=15000)
    parser.add_argument('--accumulate_grad_batches', type=int, help='Number of batches to accumulate', default=1)
    parser.add_argument('--loss', type=str, help='Loss function', default='mse')
    parser.add_argument('--lam_l', type=float, help='Lambda similarity weight', default=1)
    parser.add_argument('--lam_s', type=float, help='Lambda segmentation weight', default=1)
    parser.add_argument('--lam_m', type=float, help='Lambda magnitude weight', default=0.001)
    parser.add_argument('--lam_g', type=float, help='Lambda gradient weight', default=0.005)
    parser.add_argument('--precision', type=int, help='Precision', default=32)
    parser.add_argument('--tensor-cores', type=bool, help='Use tensor cores', default=False)
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--inshape', type=int, help='Input shape', default=128)
    parser.add_argument('--num_inter_by_epoch', type=int, help='Number of interpolations by epoch', default=1)
    parser.add_argument('--mode', type=str, help='SVF Temporal mode', choices={'mlp', 'linear'}, default='linear')
    parser.add_argument('--mlp_hidden_size', type=int, nargs='+', help='Hidden size of the MLP model', default=[1, 32, 32, 32, 1])
    parser.add_argument('--load', type=str, help='Load registration model', default='')
    parser.add_argument('--load_mlp', type=str, help='Load MLP model', default='')

    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    train_main(args=args)
