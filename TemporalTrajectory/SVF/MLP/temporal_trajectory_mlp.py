import itertools

import numpy as np
import torch
import torchio as tio

from torch import Tensor
from utils.loss import *
import pytorch_lightning as pl
from Registration import RegistrationModuleSVF, SpatialTransformer, VecInt
from utils import dice_score
import random
import matplotlib.pyplot as plt
from network import MLP

class TemporalTrajectoryMLPSVF(pl.LightningModule):
    def __init__(self, reg_model: RegistrationModuleSVF, mlp_model : MLP, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 3, save_path: str = ""):
        super().__init__()
        self.reg_model = reg_model
        self.temporal_mlp = mlp_model
        self.sim_loss = GetLoss(loss)
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.mse_loss = nn.MSELoss()
        self.subject_t0 = None
        self.subject_t1 = None
        self.num_classes = num_classes
        self.save_path = save_path


    def forward(self, source: Tensor, target: Tensor, age: Tensor):
        velocity = self.reg_model(source, target)
        coef = self.temporal_mlp(age)
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
        return forward_flow, backward_flow


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.reg_model.train()
        self.temporal_mlp.train()


    def on_train_start(self) -> None:
        self.counter_samples_indexes = np.zeros(len(self.trainer.train_dataloader.dataset.dataset))
        self.dice_max = 0
        self.num_inter_by_epoch = 10
        for i in range(len(self.trainer.train_dataloader.dataset.dataset)):
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 0:
                self.subject_t0 = self.trainer.train_dataloader.dataset.dataset[i]
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 1:
                self.subject_t1 = self.trainer.train_dataloader.dataset.dataset[i]
        self.subject_t0['image'][tio.DATA] = self.subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t0['label'][tio.DATA] = self.subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['image'][tio.DATA] = self.subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['label'][tio.DATA] = self.subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.trainer.loggers[0].experiment.add_graph(self, (self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA], torch.asarray([1.0]).to(self.device)))

    def training_step(self, batch, batch_idx):
        loss_intermediate_tensor = torch.zeros((4)).to(self.device).float()
        velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
        coef = self.temporal_mlp(torch.asarray([self.subject_t1['age']]).to(self.device))
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
        loss_primary_registration = self.reg_model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            self.counter_samples_indexes[i+1] += 1
            subject = self.trainer.train_dataloader.dataset.dataset[i+1]
            coef = self.temporal_mlp(torch.asarray([subject['age']]).to(self.device))
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
            loss_intermediate_tensor += self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=0, lambda_grad=0)
        loss_array = loss_primary_registration + loss_intermediate_tensor
        loss = loss_array[0] + loss_array[1] + loss_array[2] + loss_array[3]
        loss += 1.0 * (self.sim_loss(self.temporal_mlp(torch.asarray([0.0]).to(self.device)), torch.asarray([0.0]).to(self.device))) # Lock the deformation to 0 at t=0
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_array[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", loss_array[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", loss_array[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", loss_array[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch % 10 == 0:
            self.reg_model.eval()
            self.temporal_mlp.eval()
            all_dice = []
            with torch.no_grad():
                velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
                for subject in self.trainer.train_dataloader.dataset.dataset:
                    coef = self.temporal_mlp(torch.asarray([subject['age']]).to(self.device))
                    forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
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
                    torch.save(self.reg_model.state_dict(), self.save_path + "/reg_model_best.pth")
                    torch.save(self.temporal_mlp.state_dict(), self.save_path + "/mlp_model_best.pth")
            plt.figure(figsize=(10, 8))
            plt.bar(range(len(self.counter_samples_indexes)), self.counter_samples_indexes, width=0.8, color="blueviolet")  # Customize bins and appearance
            self.trainer.loggers[0].experiment.add_figure("Sample dataset distribution", plt.gcf(), global_step=0)
            plt.close()

            x = np.arange(0, 1, 0.01)
            y = []
            x_label = []
            for i in x:
                coef = self.temporal_mlp(torch.asarray([i]).float().to(self.device))
                y.append(coef.detach().cpu().numpy())
                x_label.append(i)
            plt.figure(figsize=(10, 8))
            plt.plot(x, y)
            self.trainer.loggers[0].experiment.add_figure("Temporal MLP coefficient",  plt.gcf(), global_step=0)
            plt.close()
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_model_best.pth")
        torch.save(self.temporal_mlp.state_dict(), self.save_path + "/last_model_best.pth")


class TemporalTrajectoryActiveLearningMLP(TemporalTrajectoryMLPSVF):
    def __init__(self, reg_model: RegistrationModuleSVF, mlp_model : MLP, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 2, save_path: str = ""):
        super().__init__(reg_model, mlp_model, loss, lambda_sim, lambda_seg, lambda_mag, lambda_grad, num_classes, save_path)
        self.subject_t0 = None
        self.subject_t1 = None
        self.index_train_current_epoch = None
        self.num_inter_by_epoch = 10

    def on_train_epoch_start(self) -> None:
        self.reg_model.train()
        self.temporal_mlp.train()


    def on_train_start(self) -> None:
        self.counter_samples_indexes = np.zeros(len(self.trainer.train_dataloader.dataset.dataset))
        self.dice_max = 0
        self.index_train_current_epoch = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in range(len(self.trainer.train_dataloader.dataset.dataset)):
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 0:
                self.subject_t0 = self.trainer.train_dataloader.dataset.dataset[i]
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 1:
                self.subject_t1 = self.trainer.train_dataloader.dataset.dataset[i]
        self.subject_t0['image'][tio.DATA] = self.subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t0['label'][tio.DATA] = self.subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['image'][tio.DATA] = self.subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['label'][tio.DATA] = self.subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.trainer.loggers[0].experiment.add_graph(self, (self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA], torch.asarray([1.0]).to(self.device)))

    def training_step(self, batch, batch_idx):
        loss_intermediate_tensor = torch.zeros((4)).to(self.device).float()
        velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
        coef = self.temporal_mlp(torch.asarray([self.subject_t1['age']]).to(self.device))
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
        loss_primary_registration = self.reg_model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            self.counter_samples_indexes[i+1] += 1
            subject = self.trainer.train_dataloader.dataset.dataset[i+1]
            coef = self.temporal_mlp(torch.asarray([subject['age']]).to(self.device))
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
            loss_intermediate_tensor += self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=0, lambda_grad=0)
        loss_array = loss_primary_registration + loss_intermediate_tensor
        loss = loss_array[0] + loss_array[1] + loss_array[2] + loss_array[3]
        loss += 1.0 * (self.sim_loss(self.temporal_mlp(torch.asarray([0.0]).to(self.device)), torch.asarray([0.0]).to(self.device))) # Lock the deformation to 0 at t=0
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_array[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", loss_array[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", loss_array[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", loss_array[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def on_train_epoch_end(self):
        self.reg_model.eval()
        self.temporal_mlp.eval()
        all_dice = []
        accumulate_errors = torch.tensor([])
        with torch.no_grad():
            velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
            for subject in self.trainer.train_dataloader.dataset.dataset:
                coef = self.temporal_mlp(torch.asarray([subject['age']]).to(self.device))
                forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
                loss = self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=0, lambda_grad=0)
                accumulate_errors = torch.cat((accumulate_errors, torch.sum(loss)))
                warped_source_label = self.reg_model.warp(self.subject_t0['label'][tio.DATA].unsqueeze(dim=0).float().to(self.device), forward_flow, mode='nearest')
                one_hot_warped_source = F.one_hot(warped_source_label.squeeze().long(),
                                                  num_classes=self.num_classes).float().permute(3, 0, 1, 2)
                one_hot_subject = F.one_hot(subject['label'][tio.DATA].squeeze().long(),
                                            num_classes=self.num_classes).permute(3, 0, 1, 2)
                dice = dice_score(one_hot_warped_source.detach().cpu().numpy(), one_hot_subject.detach().cpu().numpy())
                mean_dice = np.mean(dice[1::])
                all_dice.append(mean_dice)


            cumulative_loss = torch.cumsum(accumulate_errors, dim=0)
            cumulative_loss_normalized = (cumulative_loss - torch.min(cumulative_loss)) / (
                        torch.max(cumulative_loss) - torch.min(cumulative_loss))
            self.index_train_current_epoch = [torch.abs(cumulative_loss_normalized - random.random()).argmin() for _ in range(self.num_inter_by_epoch)]

            mean_all_dice = np.mean(all_dice)
            self.log("Mean dice", mean_all_dice)
            if mean_all_dice > self.dice_max:
                self.dice_max = mean_all_dice
                print("New best dice:", self.dice_max)
                torch.save(self.reg_model.state_dict(), self.save_path + "/reg_model_best.pth")
                torch.save(self.temporal_mlp.state_dict(), self.save_path + "/mlp_model_best.pth")

        if self.current_epoch % 100 == 0:
            plt.figure(figsize=(10, 8))
            plt.bar(range(len(self.counter_samples_indexes)), self.counter_samples_indexes, width=0.8, color="blueviolet")  # Customize bins and appearance
            self.trainer.loggers[0].experiment.add_figure("Sample dataset distribution", plt.gcf(), global_step=0)
            plt.close()
            x = np.arange(0, 1, 0.01)
            y = []
            x_label = []
            for i in x:
                coef = self.temporal_mlp(torch.asarray([i]).float().to(self.device))
                y.append(coef.detach().cpu().numpy())
                x_label.append(i)
            plt.figure(figsize=(10, 8))
            plt.plot(x, y)
            self.trainer.loggers[0].experiment.add_figure("Temporal MLP coefficient",  plt.gcf(), global_step=0)
            plt.close()