import os
import sys
import torch
import monai
import argparse
import numpy as np
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from monai.metrics import DiceMetric
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, ".")
sys.path.insert(1, "..")

import torchio2 as tio
from utils.loss import GetLoss
from dataset import PairwiseSubjectsDataset
from registration_module import RegistrationModuleSVF, RegistrationModule
from utils import create_directory, write_namespace_arguments


class RegistrationTrainingModule(pl.LightningModule):
    def __init__(self, model : RegistrationModuleSVF | RegistrationModule, lambda_sim: float = 1.0, lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, save_path: str = "./"):
        super().__init__()
        self.model = model
        self.sim_loss = GetLoss(args.loss)
        self.seg_loss = monai.losses.GeneralizedDiceLoss(include_background=False)
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.save_path = save_path
        self.dice_metric = DiceMetric(include_background=True, reduction="none")
        self.dice_max = 0

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        return self.model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.dice_max = 0
        self.model.train()

    def training_step(self, batch):
        '''
            Compute the loss of the model on each pair of images
        '''
        source, target = batch.values()
        forward_flow, backward_flow = self.model.forward_backward_flow_registration(source['image'][tio.DATA], target['image'][tio.DATA])
        loss_tensor = self.registration_loss(source, target, forward_flow, backward_flow)
        loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0] * self.lambda_sim, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def registration_loss(self, source: tio.Subject, target: tio.Subject, forward_flow: Tensor, backward_flow: Tensor) -> Tensor:
        '''
            Compute the registration loss for a pair of subjects
        '''
        '''
            Compute the registration loss for a pair of subjects
        '''
        loss_pair = torch.zeros((4)).float().to(self.device)
        target_image = target['image'][tio.DATA].to(self.device)
        source_image = source['image'][tio.DATA].to(self.device)

        if len(target_image.shape) == 4:
            target_image = target_image.unsqueeze(dim=0)
        if len(source_image.shape) == 4:
            source_image = source_image.unsqueeze(dim=0)

        if self.lambda_sim > 0:
            loss_pair[0] =  (self.sim_loss(self.model.warp(source_image, forward_flow), target_image) +
                             self.sim_loss(self.model.warp(target_image, backward_flow), source_image))

        if self.lambda_seg > 0:
            target_label = target['label'][tio.DATA].float().to(self.device)
            source_label = source['label'][tio.DATA].float().to(self.device)
            if len(target_label.shape) == 4:
                target_label = target_label.unsqueeze(dim=0)
            if len(source_label.shape) == 4:
                source_label = source_label.unsqueeze(dim=0)
            warped_source_label = self.model.warp(source_label.float(), forward_flow)
            warped_target_label = self.model.warp(target_label.float(), backward_flow)
            loss_pair[1] = nn.MSELoss()(warped_source_label[:,1:, ...], target_label[:,1:, ...]) + nn.MSELoss()(warped_target_label[:,1:, ...], source_label[:,1:, ...])
        if self.lambda_mag > 0:
            loss_pair[2] = nn.MSELoss()(forward_flow, torch.zeros(forward_flow.shape, device=self.device)) + nn.MSELoss()(backward_flow, torch.zeros(backward_flow.shape, device=self.device))

        if self.lambda_grad > 0:
            loss_pair[3] = self.model.regularizer(forward_flow, penalty='l2').to(self.device) + self.model.regularizer(backward_flow, penalty='l2').to(self.device)
        return loss_pair

    def on_train_epoch_end(self):
        '''
            Compute the dice score on the training dataset
        '''
        torch.save(self.model.state_dict(), self.save_path + "/last_model.pth")

    def validation_step(self, batch):
        self.dice_metric.reset()
        source, target = batch.values()
        dice_scores = []
        with torch.no_grad():
            forward_flow, backward_flow = self.model.forward_backward_flow_registration(source['image'][tio.DATA], target['image'][tio.DATA])
            warped_source_label = self.model.warp(
                torch.round(source['label'][tio.DATA].float().to(self.device)), forward_flow)
            self.dice_metric(torch.round(warped_source_label), target['label'][tio.DATA].float().to(self.device))
            loss_tensor = self.registration_loss(source, target, forward_flow, backward_flow)
            loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()
            dice = self.dice_metric(torch.argmax(warped_source_label), target['label'][tio.DATA].float().to(self.device))[0]
            dice_scores.append(torch.mean(dice[1:]).cpu().numpy())
            self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("Similitude", self.lambda_sim * loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)

        mean_dices =  sum(dice_scores) / len(dice_scores)
        self.log("Mean dice", mean_dices, prog_bar=True, on_epoch=True, sync_dist=True)
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.model.state_dict(), self.save_path + "/best_model.pth")
        return loss


def train(args):
    ## Config Dataset / Dataloader
    train_transforms = tio.Compose([
        tio.CropOrPad(target_shape=args.inshape),
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.OneHot(args.num_classes)
    ])

    ## Dateset's configuration : Load a pairwise dataset and the dataloader
    dataset = PairwiseSubjectsDataset(dataset_path=args.csv, transform=train_transforms, age=False)
    loader = tio.SubjectsLoader(dataset, batch_size=args.batch_size, num_workers=15)
    in_shape = dataset[0]['0']['image'][tio.DATA].shape[1:]

    ## Model initialization and weights loading if needed
    try:
        model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32, 64], strides=[2,2,2]), inshape=in_shape, int_steps=7)
        if args.load != "":
            model.load_state_dict(torch.load(args.load))
    except:
        raise ValueError("Model initialization failed")

    ## Config training with hyperparameters
    trainer_args = {
        'epochs': args.epochs,
        'max_steps': args.max_steps,
        'precision': args.precision,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./log", name=None),
        'check_val_every_n_epoch': 20,
        'num_sanity_val_steps': 0,
        'enable_progress_bar': True if args.progress_bar is True else False,
    }

    if args.multi_gpu:
        trainer_args['accelerator'] = 'gpu'
        trainer_args['strategy'] = 'ddp'
        trainer_args['devices'] = int(os.environ['SLURM_GPUS_ON_NODE'])
        trainer_args['num_nodes'] = int(os.environ['SLURM_NNODES'])

    save_path = trainer_args['logger'].log_dir.replace("log", "Results")

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=1000,  # Save every 1000 steps
    )
    trainer_args['callbacks'] = [checkpoint_callback]
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "config.json"))

    # Train the model
    training_module = RegistrationTrainingModule(
        model=model,
        lambda_sim=args.lam_l,
        lambda_seg=args.lam_s,
        lambda_mag=args.lam_m,
        lambda_grad=args.lam_g,
        save_path=save_path
    )

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(training_module, train_dataloaders=loader, val_dataloaders=loader)
    torch.save(training_module.model.state_dict(), os.path.join(save_path + "/final_model_reg.pth"))


# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Registration 3D Images')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='../data/full_dataset.csv')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('--max_steps', type=int, help='Number of steps', default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, help='Number of batches to accumulate', default=4)
    parser.add_argument('--loss', type=str, help='Loss function', default='mse')
    parser.add_argument('--lam_l', type=float, help='Lambda similarity weight', default=100)
    parser.add_argument('--lam_s', type=float, help='Lambda segmentation weight', default=200)
    parser.add_argument('--lam_m', type=float, help='Lambda magnitude weight', default=0.01)
    parser.add_argument('--progress_bar', type=bool, help='Precision', default=True)
    parser.add_argument('--lam_g', type=float, help='Lambda gradient weight', default=0.01)
    parser.add_argument('--precision', type=int, help='Precision', default=32)
    parser.add_argument('--inshape', type=int, nargs='+', help='Input shape', default=[128, 128, 128])
    parser.add_argument('--tensor-cores', type=bool, help='Use tensor cores', default=False)
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--inshape', type=int, help='Input shape', default=128)
    parser.add_argument('--load', type=str, help='Load registration model', default='')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--multi_gpu', type=bool, help='Use multi-gpu', default=False)
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    train(args=args)
