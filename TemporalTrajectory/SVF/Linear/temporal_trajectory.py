import itertools

import torch
import torchio as tio
from torch import Tensor
from utils.loss import *
import monai.networks.nets
import pytorch_lightning as pl
import torch.nn.functional as F
from Registration import RegistrationModuleSVF
from utils import dice_score
import random

class TemporalTrajectorySVF(pl.LightningModule):
    def __init__(self, reg_model: RegistrationModuleSVF, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__()
        self.reg_model = reg_model
        self.sim_loss = GetLoss(loss)
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.mDice = 0
        self.pairwise_index = [[0, 1, 2], [0, 2, 1], [1, 2, 0]]

    def on_train_epoch_start(self) -> None:
        self.reg_model.train()

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
        self.dice_max = 0
        self.num_inter_by_epoch = 10

    def forward(self, source: Tensor, target: Tensor):
        return self.reg_model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):

        loss_intermediate_tensor = torch.zeros((4)).to(self.device).float()
        velocity = self(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity)
        loss_primary_registration = self.reg_model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            subject = self.trainer.train_dataloader.dataset.dataset[i+1]
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * subject['age'])
            loss_intermediate_tensor += self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=0, lambda_grad=0)
        loss_array = loss_primary_registration + loss_intermediate_tensor
        loss = torch.sum(loss_array)
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", self.lambda_sim * loss_array[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_array[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_array[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_array[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            self.reg_model.eval()
            all_dice = []
            with torch.no_grad():
                velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
                for subject in self.trainer.train_dataloader.dataset.dataset:
                    forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * subject['age'])
                    warped_source_label = self.reg_model.warp(self.subject_t0['label'][tio.DATA].unsqueeze(dim=0).float().to(self.device), forward_flow, mode='nearest')
                    one_hot_warped_source = F.one_hot(warped_source_label.squeeze().long(), num_classes=self.num_classes).float().permute(3, 0, 1, 2)
                    one_hot_subject = F.one_hot(subject['label'][tio.DATA].squeeze().long(), num_classes=self.num_classes).permute(3, 0, 1, 2)
                    dice = dice_score(one_hot_warped_source.detach().cpu().numpy(), one_hot_subject.detach().cpu().numpy())
                    mean_dice = np.mean(dice[1::])
                    all_dice.append(mean_dice)
                mean_all_dice = np.mean(all_dice)
                self.log("Mean dice", mean_all_dice)
                if mean_all_dice > self.dice_max:
                    self.dice_max = mean_all_dice
                    print("New best dice:", self.dice_max)
                    torch.save(self.reg_model.state_dict(), "./model_linear_best.pth")
        torch.save(self.reg_model.state_dict(), "./last_model_reg.pth")





        self.reg_model.eval()
        all_dice = []
        with torch.no_grad():
            velocity = self(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])

            for subject in self.trainer.train_dataloader.dataset.dataset:
                forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * subject['age'])
                warped_source_label = self.reg_model.warp(self.subject_t0['label'][tio.DATA].float().to(self.device), forward_flow)
                dice = dice_score(torch.argmax(warped_source_label, dim=1), torch.argmax(subject['label'][tio.DATA].float().to(self.device), dim=0), num_classes=20)
                mean_dice = np.mean(dice)
                all_dice.append(mean_dice)
            mean_all_dice = np.mean(all_dice)
            self.log("Mean dice", mean_all_dice)
            if mean_all_dice > self.dice_max:
                self.dice_max = mean_all_dice
                torch.save(self.reg_model.state_dict(), "./model_linear_best.pth")



class LongitudinalAtlasMeanVelocityModule(TemporalTrajectorySVF):
    def __init__(self, reg_model: RegistrationModuleSVF, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__(reg_model=reg_model, loss=loss, lambda_sim=lambda_sim, lambda_seg=lambda_seg, lambda_mag=lambda_mag, lambda_grad=lambda_grad)
        self.mDice = 0
        self.weightedMeanVelocity = None

    def on_train_epoch_end(self):
        torch.save(self.reg_model.state_dict(), "./model_last_epoch.pth")
        torch.save(self.weightedMeanVelocity, "./weightedMeanVelocityTensor.pth")

    def training_step(self, batch, batch_idx):
        num = torch.zeros([1, 3] + list(self.shape)).to(self.device)
        denum = 0
        for i, j in self.index_pairs:
            samples_i = self.trainer.train_dataloader.dataset.get_subject(i)
            samples_j = self.trainer.train_dataloader.dataset.get_subject(j)
            time_ij = samples_j['age'] - samples_i['age']
            velocity_ij = self(samples_i['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device), samples_j['image'][tio.DATA].unsqueeze(dim=0).float().to(self.device))
            num += velocity_ij * time_ij
            denum += time_ij * time_ij
        weightedMeanVelocity = num / denum if denum != 0 else torch.zeros_like(num)

        loss_tensor = torch.zeros((4)).to(self.device).float()
        for i in range(len(self.trainer.train_dataloader.dataset.subject_dataset)):
            target_subject = self.trainer.train_dataloader.dataset.get_subject(i)
            weight = target_subject['age'] - batch['age'][0]
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(weightedMeanVelocity * weight)
            loss_tensor += self.reg_model.registration_loss(batch, target_subject, forward_flow, backward_flow)

        loss = self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", self.lambda_sim * loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


