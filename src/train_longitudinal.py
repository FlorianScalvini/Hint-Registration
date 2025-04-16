import os
import random
import torch
import torch.nn as nn
from torch import Tensor
import torchio as tio
import pytorch_lightning as pl
from losses import PairwiseRegistrationLoss
from modules.longitudinal_deformation import OurLongitudinalDeformation
from monai.metrics import DiceMetric

class LongitudinalTrainingModule(pl.LightningModule):
    '''
        Lightning Module to train a Longitudinal Estimation of Deformation
    '''
    def __init__(self, model: OurLongitudinalDeformation, loss: PairwiseRegistrationLoss, learning_rate: float = 0.001,
                 save_path: str = "./", num_inter_by_epoch=1, penalize: str = 'v'):
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
        self.dice_metric = DiceMetric(include_background=True, reduction="none")
        self.dice_max = 0 # Maximum dice score
        self.learning_rate = learning_rate

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

    def forward(self, source: Tensor, target: Tensor):
        return self.model((source, target))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, _):
        velocity = self.model.forward((self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA]))
        index = random.sample(range(1, self.trainer.train_dataloader.dataset.num_subjects), self.num_inter_by_epoch)
        image_loss = torch.zeros(1).to(self.device)
        flow_loss = torch.zeros(1).to(self.device)

        for i in index:
            k = self.trainer.train_dataloader.dataset[i]
            time = self.model.encode_time(torch.Tensor([k['age']]).to(self.device))
            flow_J = self.model.reg_model.velocity2displacement(time * velocity)
            Jw = self.model.reg_model.warp(self.subject_t0['label'][tio.DATA].float(), flow_J)
            flow_I = self.model.reg_model.velocity2displacement((time - 1.) * velocity)
            Iw = self.model.reg_model.warp(self.subject_t1['label'][tio.DATA].float(), flow_I)
            K = k['label'][tio.DATA].float().to(Jw.device).unsqueeze(dim=0)


            image_loss +=  (torch.nn.MSELoss()(Jw, Iw) +
                           torch.nn.MSELoss()(K, Iw) +
                           torch.nn.MSELoss()(K, Jw)) / 3.0


            flow_2t_1 = self.model.reg_model.velocity2displacement(((2. * time) - 1.) * velocity)
            flow_JI = self.model.reg_model.warp(flow_J, flow_I) + flow_I
            flow_IJ = self.model.reg_model.warp(flow_I, flow_J) + flow_J
            flow_loss += 0.5 * (torch.mean((flow_2t_1 - flow_JI) ** 2) + torch.mean((flow_2t_1 - flow_IJ) ** 2))

        loss = (image_loss + 0.5 * flow_loss + nn.MSELoss()(torch.zeros(1).to(self.device), self.model.encode_time(torch.zeros(1).to(self.device))) +
                nn.MSELoss()(torch.ones(1).to(self.device), self.model.encode_time(torch.ones(1).to(self.device))))
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", image_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Regu", flow_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

        """
        inter_image = intermediate_subject['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        inter_label = intermediate_subject['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        losses += self.loss(self.subject_t0['image'][tio.DATA], inter_image, self.subject_t0['label'][tio.DATA],
                            inter_label, forward_flow, backward_flow, None if self.penalize == 'd' else velocity)
        
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", losses[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", losses[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", losses[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", losses[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Inverse consistency", losses[4], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
        """

    def on_train_epoch_end(self):
        torch.save(self.model.reg_model.state_dict(), self.save_path + "/last_model_reg.pth")
        if self.model.time_mode == 'mlp':
            torch.save(self.model.temp_model.state_dict(), self.save_path + "/last_model_mlp.pth")


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
            for subject in self.trainer.val_dataloaders.dataset:
                timed_velocity = self.model.encode_time(torch.Tensor([subject['age']]).to(self.device)) * velocity
                forward_flow = self.model.reg_model.velocity2displacement(timed_velocity)
                warped_source_label = self.model.reg_model.warp(subject_t0['label'][tio.DATA].to(self.device).float(),
                                                      forward_flow)
                self.dice_metric(torch.nn.functional.one_hot(torch.argmax(warped_source_label, dim=1), num_classes=warped_source_label.size(1)).permute(0,4,1,2,3),
                                 subject['label'][tio.DATA].to(self.device).int().unsqueeze(0))
                overall_dice = self.dice_metric.aggregate()
                self.dice_metric.reset()
        timed_velocity = velocity
        forward_flow = self.model.reg_model.velocity2displacement(timed_velocity)
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
        tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu().numpy(),
                        affine=subject_t0['image'].affine).save(
            self.save_path + "/forward_dvf.nii.gz")
        mean_dices = torch.mean(overall_dice).item()
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.model.reg_model.state_dict(), self.save_path + "/model_reg_best.pth")
            if self.model.temp_model is not None:
                torch.save(self.model.temp_model.state_dict(), self.save_path + "/model_mlp_best.pth")
            self.log("Dice max", self.dice_max, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Mean dice", mean_dices, prog_bar=True, on_epoch=True, sync_dist=True)

    def save(self, path: str):
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.reg_model.state_dict(), os.path.join(path, "reg_model.pth"))
        if self.model.mode == 'mlp':
            torch.save(self.mlp_model.state_dict(), os.path.join(path, "temporal_model.pth"))
