import itertools
import torchio as tio
from torch import Tensor
from utils.loss import *
import monai.networks.nets
import pytorch_lightning as pl
import torch.nn.functional as F
from Registration import RegistrationModuleSVF


class TemporalTrajectorySVF(pl.LightningModule):
    def __init__(self, model: RegistrationModuleSVF, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__()
        self.model = model
        self.sim_loss = GetLoss(loss)
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.mDice = 0
        self.pairwise_index = [[0, 1, 2], [0, 2, 1], [1, 2, 0]]
    def forward(self, source: Tensor, target: Tensor):
        return self.model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss_tensor = torch.zeros((4)).to(self.device).float()
        for i,j,k in enumerate(self.pairwise_index):
            sample_i = batch[i]
            sample_j = batch[j]
            sample_k = batch[k]
            velocity_ij = self(sample_i['image'][tio.DATA].float(), sample_j['image'][tio.DATA].float())
            forward_flow, backward_flow = self.model.velocity_to_flow(velocity_ij)
            loss_tensor += self.model.registration_loss(sample_i, sample_j, forward_flow, backward_flow)

            weight = (sample_k['age'][0] - sample_j['age'][0]) / (sample_j['age'][0] - sample_i['age'][0])

            velocity_bis = velocity_ij * weight
            forward_flow, backward_flow = self.model.velocity_to_flow(velocity_bis)
            loss_tensor += self.model.registration_loss(sample_j, sample_k, forward_flow, backward_flow)
            weight = (sample_k['age'][0] - sample_j['age'][0]) / (sample_j['age'][0] - sample_i['age'][0])

            velocity_bis = velocity_ij * weight
            forward_flow, backward_flow = self.model.velocity_to_flow(velocity_bis)
            loss_tensor += self.model.registration_loss(sample_i, sample_k, forward_flow, backward_flow)

        loss = (loss_tensor[0] * self.lambda_sim + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2]
                + self.lambda_grad * loss_tensor[3]).float()

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", self.lambda_sim * loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), "./model_last_epoch.pth")


class LongitudinalAtlasMeanVelocityModule(LongitudinalAtlasModule):
    def __init__(self, model: RegistrationModuleSVF, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__(model=model, loss=loss, lambda_sim=lambda_sim, lambda_seg=lambda_seg, lambda_mag=lambda_mag, lambda_grad=lambda_grad)
        self.mDice = 0
        self.weightedMeanVelocity = None

    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), "./model_last_epoch.pth")
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
            forward_flow, backward_flow = self.model.velocity_to_flow(weightedMeanVelocity * weight)
            loss_tensor += self.model.registration_loss(batch, target_subject, forward_flow, backward_flow)

        loss = self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", self.lambda_sim * loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


