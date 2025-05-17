import os
import random
import monai.losses
from torch import Tensor
import torchio as tio
import pytorch_lightning as pl
from losses import PairwiseRegistrationLoss
from modules.longitudinal_deformation import OurLongitudinalDeformation
from monai.metrics import DiceMetric
import torch
import losses
import losses.similarity
from utils.grid_utils import compose, warp, flow2phi
from utils.visualize import plt_grid
from torchvision import transforms

class LongitudinalTrainingModule(pl.LightningModule):
    '''
        Lightning Module to train a Longitudinal Estimation of Deformation
    '''
    def __init__(self, model: OurLongitudinalDeformation, loss: PairwiseRegistrationLoss, learning_rate: float = 0.001,
                 save_path: str = "./", num_inter_by_epoch=1, penalize: str = 'v', lambda_reg = 0.05):
        '''
        :param model: Registration model
        :param loss: PairwiseRegistrationLoss function
        :param save_path: Path to save the model
        :param num_inter_by_epoch: Number of time points by epoch
        '''
        super().__init__()
        self.loss = loss
        if penalize not in ['v', 'd']:
            raise ValueError("Penalize must be 'v' or 'd'")
        self.penalize = penalize
        self.model = model
        self.save_path = save_path
        self.num_inter_by_epoch = num_inter_by_epoch
        self.dice_metric = DiceMetric(include_background=True, reduction="sum")
        self.dice_max = 0 # Maximum dice score
        self.learning_rate = learning_rate
        self.loss = monai.losses.DiceLoss()
        self.loss_bend = monai.losses.BendingEnergyLoss(normalize=True, reduction="mean")
        self.jac_loss = losses.jacobian.Jacobianloss()
        self.lambda_reg = lambda_reg

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
        #self.all_images = torch.zeros((1, len(self.trainer.train_dataloader.dataset), 128, 128, 128)).to(self.device)
        #for i in range(len(self.trainer.train_dataloader.dataset)):
        #    self.all_images[:, i, :, :, :] = self.trainer.train_dataloader.dataset[i]['image'][tio.DATA].unsqueeze(0).to(self.device)

    def forward(self, source: Tensor, target: Tensor):
        return self.model((source, target))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, _):
        seg_loss = torch.zeros(1).to(self.device)
        reg_loss = torch.zeros(1).to(self.device)
        mag_loss = torch.zeros(1).to(self.device)
        velocity = self.model.forward(self.subject_t0['image'][tio.DATA].float())
        index = random.sample(range(self.trainer.train_dataloader.dataset.num_subjects), self.num_inter_by_epoch)
        for i in index:
            subject = self.trainer.train_dataloader.dataset[i]
            subject['label'][tio.DATA] = subject['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
            v_J = self.model.encode_time(torch.Tensor([subject['age']]).to(self.device)) * velocity
            v_I = self.model.encode_time(torch.Tensor([subject['age'] - 1.]).to(self.device)) * velocity
            flow_J = self.model.reg_model.velocity2displacement(v_J)
            flow_I = self.model.reg_model.velocity2displacement(v_I)
            Jw = warp(self.subject_t0['label'][tio.DATA].float(), flow_J)
            Iw = warp(self.subject_t1['label'][tio.DATA].float().to(self.device), flow_I)
            K = subject['label'][tio.DATA].float().to(Jw.device)
            seg_loss += (self.loss(Jw, Iw) + self.loss(K, Iw) + self.loss(K, Jw)) / 3.0
            flow_VI = warp(velocity, flow_I) + flow_I
            flow_VbJ = warp(-velocity, flow_J) + flow_J
            reg_loss += torch.mean((flow_VI - flow_J) ** 2) + torch.mean((flow_VbJ - flow_I) ** 2)
            mag_loss += 0.5*(self.loss_bend(v_J) + self.loss_bend(v_I))

        seg_loss *= 1
        reg_loss *= 0.07
        mag_loss *= 0.00001
        loss =  seg_loss + reg_loss + mag_loss
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", seg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Regu", reg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Bend", mag_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), self.save_path + "/last_model_reg.pth")


    def validation_step(self, batch, batch_idx):
        subject_t0 = None
        rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label')
        for i in range(self.trainer.val_dataloaders.dataset.num_subjects):
            subject = self.trainer.val_dataloaders.dataset[i]
            if subject['age'] == 0:
                subject_t0 = subject
        subject_t0_rescale = rescale(subject_t0)

        subject_t0['image'][tio.DATA] = subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t0['label'][tio.DATA] = subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        subject_t0_rescale['image'][tio.DATA] = subject_t0_rescale['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            velocity = self.model.forward(subject_t0_rescale['image'][tio.DATA].float())
            for subject in self.trainer.val_dataloaders.dataset:
                timed_velocity = self.model.encode_time(torch.Tensor([subject['age']]).to(self.device)) * velocity
                forward_flow = self.model.reg_model.velocity2displacement(timed_velocity)
                warped_source_label = warp(subject_t0['label'][tio.DATA].to(self.device).float(), forward_flow, mode='nearest')
                self.dice_metric(
                    torch.nn.functional.one_hot(warped_source_label.squeeze().long()).permute(3, 0, 1, 2).unsqueeze(0),
                    torch.nn.functional.one_hot(subject['label'][tio.DATA].squeeze().long()).permute(3, 0, 1, 2).to(
                        self.device).unsqueeze(0))
        forward_flow = self.model.reg_model.velocity2displacement(velocity)
        label_warped_source = warp(subject_t0['label'][tio.DATA].to(self.device).float(), forward_flow, mode='nearest')
        image_warped_source = warp(subject_t0['image'][tio.DATA].to(self.device).float(), forward_flow)
        tio.LabelMap(tensor=label_warped_source.squeeze(0).detach().cpu().numpy(), affine=subject_t0['label'].affine).save(
            self.save_path + "/label_warped_source.nii.gz")
        tio.ScalarImage(tensor=image_warped_source.squeeze(0).detach().cpu().numpy(),
                        affine=subject_t0['image'].affine).save(
            self.save_path + "/image_warped_source.nii.gz")
        tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu().numpy(),
                        affine=subject_t0['image'].affine).save(
            self.save_path + "/forward_dvf.nii.gz")
        tio.ScalarImage(tensor=(-forward_flow).squeeze(0).detach().cpu().numpy(),
                        affine=subject_t0['image'].affine).save(
            self.save_path + "/backward_dvf.nii.gz")
        overall_dice = self.dice_metric.aggregate(reduction="none")
        self.dice_metric.reset()
        mean_dices = torch.mean(overall_dice).item()
        det_j = (losses.jacobian.jacobian_determinant_3d(flow2phi(forward_flow, grid_normalize=True)) < 0).sum()
        det_j_2 = (losses.jacobian.compute_jacobian_determinant(flow2phi(forward_flow, grid_normalize=False)) < 0).sum()
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.model.state_dict(), self.save_path + "/model_reg_best.pth")
            self.log("Dice max", self.dice_max, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Mean dice", mean_dices, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Negative Jacobian", det_j_2.float(), prog_bar=True, on_epoch=True)
        self.log("Negative Jacobian_2", det_j.float(), prog_bar=True, on_epoch=True)
        xyz = flow2phi(forward_flow, grid_normalize=True)
        _, D, H, W, _ = xyz.shape
        xy = torch.cat(
            [xyz[0, :, :, D // 2, 2].unsqueeze(-1), xyz[0, :, :, D // 2, 1].unsqueeze(-1)],
            dim=-1)
        plt = plt_grid(xy=xy.cpu().numpy(), factor=1)
        plt.savefig(self.save_path + "/grid.png")


    def save(self, path: str):
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.model.reg_model.state_dict(), os.path.join(path, "reg_model.pth"))
        if self.model.mode == 'mlp':
            torch.save(self.mlp_model.state_dict(), os.path.join(path, "temporal_model.pth"))
