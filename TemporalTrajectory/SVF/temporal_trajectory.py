import random
import torchio as tio
from torch import Tensor
from utils.loss import *
import pytorch_lightning as pl
from Registration import RegistrationModuleSVF
from monai.metrics import DiceMetric


class TemporalTrajectorySVF(pl.LightningModule):
    def __init__(self, reg_model: RegistrationModuleSVF, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, save_path: str = "./", num_classes: int = 3, num_inter_by_epoch=1):
        super().__init__()
        self.reg_model = reg_model
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
        self.reg_model.train()

    def on_validation_epoch_start(self):
        self.reg_model.eval()

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
        return self.reg_model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity)
        loss_tensor = self.reg_model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)

        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            subject = self.trainer.train_dataloader.dataset.dataset[i + 1]
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity * subject['age'])
            loss_tensor += self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def on_train_epoch_end(self):
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_model_reg.pth")

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
            velocity = self.reg_model(subject_t0['image'][tio.DATA], subject_t1['image'][tio.DATA])
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity)
            label_warped_source = self.reg_model.warp(subject_t0['label'][tio.DATA].to(self.device).float(),
                                                   forward_flow)
            image_warped_source = self.reg_model.warp(subject_t0['image'][tio.DATA].to(self.device).float(),
                                                   forward_flow)
            tio.LabelMap(tensor=torch.argmax(label_warped_source, dim=1).int().detach().cpu().numpy(),
                         affine=subject_t0['label'].affine).save(
                self.save_path + "/label_warped_source.nii.gz")
            tio.ScalarImage(tensor=image_warped_source.squeeze(0).detach().cpu().numpy(),
                            affine=subject_t0['image'].affine).save(
                self.save_path + "/image_warped_source.nii.gz")
            for subject in self.trainer.val_dataloaders.dataset.dataset:
                forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * subject['age'])
                warped_source_label = self.reg_model.warp(subject_t0['label'][tio.DATA].to(self.device).float(),
                                                      forward_flow)
                self.dice_metric(torch.round(warped_source_label).int(),
                                 subject['label'][tio.DATA].to(self.device).int().unsqueeze(0))
            overall_dice = self.dice_metric.aggregate()
            self.dice_metric.reset()
            mean_dices = torch.mean(overall_dice).item()
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            print("New best dice:", self.dice_max)
            torch.save(self.reg_model.state_dict(), self.save_path + "/model_linear_best.pth")

    def save_model(self, path: str):
        torch.save(self.reg_model.state_dict(), path)

    def load_model(self, path: str):
        self.reg_model.load_state_dict(torch.load(path))

