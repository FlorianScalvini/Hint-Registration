#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import monai
import argparse
import torchio as tio
import pytorch_lightning as pl
from sympy.physics.units import velocity

sys.path.insert(0, ".")
from network import MLP
from dataset import OneWrappedSubjectDataset
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils import create_directory, write_namespace_arguments
from LongitudinalDeformation import OurLongitudinalDeformation
import random
from torch import Tensor
from utils.loss import *
from Registration import RegistrationModuleSVF
from monai.metrics import DiceMetric

class LongDeformTrainPL(pl.LightningModule):
    def __init__(self, model: OurLongitudinalDeformation, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, save_path: str = "./", num_classes: int = 3, num_inter_by_epoch=1):
        super().__init__()
        self.model = model
        self.sim_loss = GetLoss(loss)
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.save_path = save_path
        self.num_classes = num_classes
        self.num_inter_by_epoch = num_inter_by_epoch
        self.dice_metric = DiceMetric(include_background=True, reduction="none")
        self.dice_max = 0 # Maximum dice score

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_validation_epoch_start(self):
        self.model.eval()

    def on_train_start(self) -> None:
        self.subject_t0 = None
        self.subject_t1 = None
        for i in range(len(self.trainer.train_dataloader.dataset.dataset)):
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 0:
                self.subject_t0 = self.trainer.train_dataloader.dataset.dataset[i]
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 1:
                self.subject_t1 = self.trainer.train_dataloader.dataset.dataset[i]
        self.subject_t0['image'][tio.DATA] = self.subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t0['label'][tio.DATA] = self.subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['image'][tio.DATA] = self.subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['label'][tio.DATA] = self.subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)

    def forward(self, source: Tensor, target: Tensor, age: Tensor):
        return self.model((source, target))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        _ = self.model.forward((self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA]))
        forward_flow, backward_flow = self.model.getDeformationFieldFromTime(1.0)
        loss_tensor = self.model.reg_model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            subject = self.trainer.train_dataloader.dataset.dataset[i + 1]
            forward_flow, backward_flow = self.model.getDeformationFieldFromTime(subject['age'])
            loss_tensor += self.model.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()

        if self.model.mode == "mlp":
            loss_zero = F.mse_loss(torch.abs(self.model.mlp_model(torch.tensor([0.0]).to(self.device))),
                                   torch.tensor([0.0]).to(self.device))
            self.log("MLP t_0", loss_zero, prog_bar=True, on_epoch=True, sync_dist=True)

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        torch.save(self.model.reg_model.state_dict(), self.save_path + "/last_model_reg.pth")
        if self.model.mode == 'mlp':
            torch.save(self.model.mlp_model.state_dict(), self.save_path + "/last_model_mlp.pth")

    def validation_step(self, batch, batch_idx):
        self.dice_metric = DiceMetric(include_background=True, reduction="none")
        subject_t0 = None
        subject_t1 = None
        for i in range(len(self.trainer.val_dataloaders.dataset.dataset)):
            if self.trainer.val_dataloaders.dataset.dataset[i]['age'] == 0:
                subject_t0 = self.trainer.val_dataloaders.dataset.dataset[i]
            if self.trainer.val_dataloaders.dataset.dataset[i]['age'] == 1:
                subject_t1 = self.trainer.val_dataloaders.dataset.dataset[i]
        subject_t0['image'][tio.DATA] = subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t0['label'][tio.DATA] = subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t1['image'][tio.DATA] = subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t1['label'][tio.DATA] = subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            velocity = self.model.forward((subject_t0['image'][tio.DATA], subject_t1['image'][tio.DATA]))
            forward_flow, backward_flow = self.model.getDeformationFieldFromTime(1.0)
            label_warped_source = self.model.reg_model.warp(subject_t0['label'][tio.DATA].to(self.device).float(),
                                                   forward_flow)
            image_warped_source = self.model.reg_model.warp(subject_t0['image'][tio.DATA].to(self.device).float(),
                                                   forward_flow)
            tio.LabelMap(tensor=torch.argmax(label_warped_source, dim=1).int().detach().cpu().numpy(),
                         affine=subject_t0['label'].affine).save(
                self.save_path + "/label_warped_source.nii.gz")
            tio.ScalarImage(tensor=image_warped_source.squeeze(0).detach().cpu().numpy(),
                            affine=subject_t0['image'].affine).save(
                self.save_path + "/image_warped_source.nii.gz")
            for subject in self.trainer.val_dataloaders.dataset.dataset:
                forward_flow, backward_flow = self.model.getDeformationFieldFromTime(subject['age'])
                warped_source_label = self.model.reg_model.warp(subject_t0['label'][tio.DATA].to(self.device).float(),
                                                      forward_flow)
                self.dice_metric(torch.round(warped_source_label).int(),
                                 subject['label'][tio.DATA].to(self.device).int().unsqueeze(0))
            overall_dice = self.dice_metric.aggregate()
            self.dice_metric.reset()
            mean_dices = torch.mean(overall_dice).item()
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            print("New best dice:", self.dice_max)
            torch.save(self.model.reg_model.state_dict(), self.save_path + "/model_reg_best.pth")
            if self.model.mode == 'mlp':
                torch.save(self.model.mlp_model.state_dict(), self.save_path + "/model_mlp_best.pth")




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
    if args.mode != 'mlp':
        model = OurLongitudinalDeformation(
            reg_model= RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32], strides=[2,2]), inshape=in_shape, int_steps=7),
            mode='linear',
            hidden_mlp_layer=None,
            t0=args.t0,
            t1=args.t1
        )
        model.load(args.load)
    else:
        model = OurLongitudinalDeformation(
            reg_model= RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32], strides=[2,2]), inshape=in_shape, int_steps=7),
            mode='mlp',
            hidden_mlp_layer=args.mlp_hidden_size,
            t0=args.t0,
            t1=args.t1
        )
        model.load(args.load, args.load_mlp)
    ## Config training
    trainer_args = {
        'max_epochs': args.epochs,
        'precision': args.precision,
        'strategy': DDPStrategy(find_unused_parameters=True),
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./log", name=None),
        'check_val_every_n_epoch': 20
    }

    save_path = trainer_args['logger'].log_dir.replace("log", args.mode + "/Results")
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "config.json"))

    training_module = LongDeformTrainPL(
        model=model,
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
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='./data/full_dataset.csv')
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
