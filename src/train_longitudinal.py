import os
import torch
import torch.nn as nn
from torch import Tensor
import random
import argparse
import monai.losses
import torchio as tio
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from .losses import GetLoss
from .utils import create_directory, write_namespace_arguments
from .modules.longitudinal_deformation import OurLongitudinalDeformation
from .modules.registration import RegistrationModuleSVF
from .modules.data import LongitudinalDataModule


class LongtitudinalLM(pl.LightningModule):
    '''
        Lightning Module to train a Longitudinal Estimation of Deformation
    '''
    def __init__(self, model: OurLongitudinalDeformation,
                 similarity_loss: str = 'mi', lambda_sim: float = 1.0,lambda_seg: float = 0, lambda_mag: float = 0,
                 lambda_grad: float = 0, save_path: str = "./", num_inter_by_epoch=1):
        '''
        :param model: Registration model
        :param similarity_loss: Loss function for the similarity
        :param lambda_sim: Weight for the similarity loss
        :param lambda_seg: Weight for the segmentation loss
        :param lambda_mag: Weight for the magnitude loss
        :param lambda_grad: Weight for the gradient loss
        :param save_path: Path to save the model
        :param num_inter_by_epoch: Number of time points by epoch
        '''
        super().__init__()
        self.model = model
        self.sim_loss = GetLoss(similarity_loss)
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.save_path = save_path
        self.num_inter_by_epoch = num_inter_by_epoch

        self.dice_max = 0 # Maximum dice score
        self.seg_loss = monai.losses.DiceCELoss(include_background=False)
        self.grad_loss = monai.losses.DiffusionLoss(normalize=True)

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_validation_epoch_start(self):
        self.model.eval()

    def on_train_start(self) -> None:
        self.subject_t0 = None
        self.subject_t1 = None
        for i in range(self.trainer.train_dataloader.dataset.num_subjects):
            subject = self.trainer.train_dataloader.dataset[i]
            if subject['age'] == 0:
                self.subject_t0 = subject
            if subject['age'] == 1:
                self.subject_t1 = subject
        self.subject_t0['image'][tio.DATA] = self.subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t0['label'][tio.DATA] = self.subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['image'][tio.DATA] = self.subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['label'][tio.DATA] = self.subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)

    def forward(self, source: Tensor, target: Tensor, age: Tensor):
        return self.model((source, target))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, _):
        velocity = self.model.forward((self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA]))
        forward_flow, backward_flow = self.model.getDeformationFieldFromTime(velocity, 1.0)
        loss_tensor = self.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow)

        index = random.sample(range(0, self.trainer.train_dataloader.dataset.num_subjects - 1), self.num_inter_by_epoch)
        for i in index:
            intermediate_subject = self.trainer.train_dataloader.dataset[i + 1]
            forward_flow, backward_flow = self.model.getDeformationFieldFromTime(velocity, intermediate_subject['age'])
            loss_tensor += self.registration_loss(self.subject_t0, intermediate_subject, forward_flow, backward_flow)
        loss_tensor[0] *= self.lambda_sim
        loss_tensor[1] *= self.lambda_seg
        loss_tensor[2] *= self.lambda_mag
        loss_tensor[3] *= self.lambda_grad
        loss = loss_tensor.sum()
        if self.model.mode == "mlp":
            loss_zero = nn.MSELoss()(self.model.mlp_model(torch.tensor([0.0]).to(self.device)), torch.tensor([0.0]).to(self.device))
            loss += loss_zero
            self.log("MLP t_0", loss_zero, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation",loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def registration_loss(self, source: tio.Subject, target: tio.Subject, forward_flow: Tensor, backward_flow: Tensor) -> Tensor:
        '''
            Compute the registration.yaml loss for a pair of subjects
        '''
        loss_pair = torch.zeros((4)).float().to(self.device)
        target_image = target['image'][tio.DATA].to(self.device)
        source_image = source['image'][tio.DATA].to(self.device)

        if len(target_image.shape) == 4:
            target_image = target_image.unsqueeze(dim=0)
        if len(source_image.shape) == 4:
            source_image = source_image.unsqueeze(dim=0)

        if self.lambda_sim > 0:
            loss_pair[0] =  (self.sim_loss(self.model.reg_model.warp(source_image, forward_flow), target_image) +
                             self.sim_loss(self.model.reg_model.warp(target_image, backward_flow), source_image))

        if self.lambda_seg > 0:
            target_label = target['label'][tio.DATA].float().to(self.device)
            source_label = source['label'][tio.DATA].float().to(self.device)
            if len(target_label.shape) == 4:
                target_label = target_label.unsqueeze(dim=0)
            if len(source_label.shape) == 4:
                source_label = source_label.unsqueeze(dim=0)
            warped_source_label = self.model.reg_model.warp(source_label.float(), forward_flow)
            warped_target_label = self.model.reg_model.warp(target_label.float(), backward_flow)
            loss_pair[1] = nn.MSELoss()(warped_source_label[:,1:, ...], target_label[:,1:, ...]) + nn.MSELoss()(warped_target_label[:,1:, ...], source_label[:,1:, ...])
        if self.lambda_mag > 0:
            loss_pair[2] = nn.MSELoss()(forward_flow, torch.zeros(forward_flow.shape, device=self.device)) + nn.MSELoss()(backward_flow, torch.zeros(backward_flow.shape, device=self.device))

        if self.lambda_grad > 0:
            loss_pair[3] = self.grad_loss(forward_flow) + self.grad_loss(backward_flow)
        return loss_pair

    def on_train_epoch_end(self):
        torch.save(self.model.reg_model.state_dict(), self.save_path + "/last_model_reg.pth")
        if self.model.mode == 'mlp':
            torch.save(self.model.mlp_model.state_dict(), self.save_path + "/last_model_mlp.pth")


    def validation_step(self, batch, batch_idx):
        subject_t0 = None
        subject_t1 = None
        for i in range(self.trainer.val_dataloaders.dataset.num_subjects):
            subject = self.trainer.val_dataloaders.dataset[i]
            if subject['age'] == 0:
                subject_t0 = subject
            if subject['age'] == 1.0:
                subject_t1 = subject
        subject_t0['image'][tio.DATA] = subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t0['label'][tio.DATA] = subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t1['image'][tio.DATA] = subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t1['label'][tio.DATA] = subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            velocity = self.model.forward((subject_t0['image'][tio.DATA], subject_t1['image'][tio.DATA]))
            forward_flow, backward_flow = self.model.getDeformationFieldFromTime(velocity, 1.0)
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
            for subject in self.trainer.val_dataloaders.dataset:
                forward_flow, backward_flow = self.model.getDeformationFieldFromTime(velocity, subject['age'])
                warped_source_label = self.model.reg_model.warp(subject_t0['label'][tio.DATA].to(self.device).float(),
                                                      forward_flow)
                self.dice_metric(torch.nn.functional.one_hot(torch.argmax(warped_source_label, dim=1), num_classes=warped_source_label.size(1)).permute(0,4,1,2,3),
                                 subject['label'][tio.DATA].to(self.device).int().unsqueeze(0))
            overall_dice = self.dice_metric.aggregate()
            self.dice_metric.reset()
            mean_dices = torch.mean(overall_dice).item()
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.model.reg_model.state_dict(), self.save_path + "/model_reg_best.pth")
            if self.model.mode == 'mlp':
                torch.save(self.model.mlp_model.state_dict(), self.save_path + "/model_mlp_best.pth")
            self.log("Dice max", self.dice_max, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Mean dice", mean_dices, prog_bar=True, on_epoch=True, sync_dist=True)


def train_main(args):

    reg_model = RegistrationModuleSVF(
        model=monai.networks.nets.AttentionUnet(
            dropout=0.1,
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            channels=[8, 16, 32, 64],
            strides=[2,2,2]),
        int_steps=7)
    model = OurLongitudinalDeformation(
        reg_model=reg_model,
        mode=args.mode,
        hidden_dim=args.mlp_hidden_dim,
        num_layers=args.mlp_num_layers,
        t0=args.t0,
        t1=args.t1,
        size=args.rsize,
    )
    if args.load != '':
        model.model.load_network(args.load)
    if args.load_temporal != '' and args.mode != 'linear':
        model.temp_model.load_state_dict()

    ## Config training
    trainer_args = {'max_steps': args.max_steps,
                    'precision': args.precision,
                    'accumulate_grad_batches': args.accumulate_grad_batches,
                    'logger': pl.loggers.TensorBoardLogger(save_dir="./" + args.mode + "/log", name=None),
                    'check_val_every_n_epoch': 20,
                    'num_sanity_val_steps': 0,
                    'callbacks': [ModelCheckpoint(every_n_train_steps=100)],
                    'log_every_n_steps': 1,
                    }
    save_path = trainer_args['logger'].log_dir.replace("log", "/Results")
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "configs.json"))

    trainer_module = LongtitudinalLM(
        model=model,
        similarity_loss=args.loss,
        lambda_sim=args.lam_l,
        lambda_seg=args.lam_s,
        lambda_mag=args.lam_m,
        lambda_grad=args.lam_g,
        save_path=save_path,
        num_inter_by_epoch=args.num_inter_by_epoch
    )
    trainer = pl.Trainer(**trainer_args)
    checkpoint = None
    if args.checkpoint != "":
        checkpoint = args.checkpoint

    datamodule = LongitudinalDataModule(
        data_dir=args.csv,
        t0=args.t0,
        t1=args.t1,
        rsize=args.rsize,
        csize=args.csize,
        batch_size=args.batch_size,
        num_workers=8
    )
    trainer.fit(trainer_module, datamodule=datamodule, ckpt_path=checkpoint)
    torch.save(trainer_module.model.reg_model.state_dict(), os.path.join(save_path, "final_reg.pth"))
    if args.mode == 'mlp' or args.mode == 'inr':
        torch.save(trainer_module.model.temporal_mlp.state_dict(), os.path.join(save_path, "final_temporal.pth"))


# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='../data/resized_dataset.csv')
    parser.add_argument('--t0', type=int, help='Initial time point', default=21)
    parser.add_argument('--t1', type=int, help='Final time point', default=36)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=15000)
    parser.add_argument('--accumulate_grad_batches', type=int, help='Number of batches to accumulate', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--loss', type=str, help='Loss function', default='lncc')
    parser.add_argument('--lam_l', type=float, help='Lambda similarity weight', default=2)
    parser.add_argument('--lam_s', type=float, help='Lambda segmentation weight', default=100)
    parser.add_argument('--lam_m', type=float, help='Lambda magnitude weight', default=0.02)
    parser.add_argument('--lam_g', type=float, help='Lambda gradient weight', default=0.5)
    parser.add_argument('--precision', type=int, help='Precision', default=32)
    parser.add_argument('--progress_bar', type=bool, help='Precision', default=True)
    parser.add_argument('--tensor-cores', type=bool, help='Use tensor cores', default=False)
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize shape', default=[128, 128, 128])
    parser.add_argument('--csize', type=int, nargs='+', help='Cropsize shape', default=[192, 192, 192])
    parser.add_argument('--num_inter_by_epoch', type=int, help='Number of interpolations by epoch', default=6)
    parser.add_argument('--mode', type=str, help='SVF Temporal mode', choices={'mlp', 'linear', 'inr'}, default='mlp')
    parser.add_argument('--mlp_num_layers', type=int, help='Number of layer of the mlp', default=4)
    parser.add_argument('--mlp_hidden_dim', type=int, help='Number of layer of the mlp', default=32)
    parser.add_argument('--load', type=str, help='Load registration.yaml model', default="")
    parser.add_argument('--load_temporal', type=str, help='Load Temporal model', default=None)
    parser.add_argument("--checkpoint", type=str, help='Path to the checkpoint', default=None)
    parser.add_argument('--histogram', type=str, help='Use histogram standardization', default=None)
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    train_main(args=args)