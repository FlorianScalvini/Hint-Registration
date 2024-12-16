
import torchio as tio
from torch import Tensor
from utils.loss import *
import pytorch_lightning as pl
from Registration import RegistrationModuleSVF
from utils import dice_score
import random
import matplotlib.pyplot as plt
from network import MLP
import torch.nn.functional as F
from monai.metrics import DiceMetric

class TemporalTrajectoryMLPSVF(pl.LightningModule):
    def __init__(self, reg_model: RegistrationModuleSVF, mlp_model : MLP, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 20, num_inter_by_epoch=4, save_path: str = ""):
        super().__init__()
        self.model = reg_model
        self.mlp = mlp_model
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
        self.num_inter_by_epoch = num_inter_by_epoch
        self.dice_metric = DiceMetric(include_background=False, reduction="none")

    def forward(self, source: Tensor, target: Tensor, age: Tensor):
        velocity = self.model(source, target)
        coef = self.mlp(age)
        forward_flow, backward_flow = self.model.velocity_to_flow(velocity * coef)
        return forward_flow, backward_flow


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.model.train()
        self.mlp.train()


    def on_train_start(self) -> None:
        self.counter_samples_indexes = np.zeros(len(self.trainer.train_dataloader.dataset.dataset))
        self.dice_max = 0
        for i in range(len(self.trainer.train_dataloader.dataset.dataset)):
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 0:
                self.subject_t0 = self.trainer.train_dataloader.dataset.dataset[i]
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 1:
                self.subject_t1 = self.trainer.train_dataloader.dataset.dataset[i]
        self.subject_t0['image'][tio.DATA] = self.subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t0['label'][tio.DATA] = self.subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['image'][tio.DATA] = self.subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['label'][tio.DATA] = self.subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.model.train()
        self.mlp.train()
        self.trainer.loggers[0].experiment.add_graph(self, (self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA], torch.asarray([1.0]).to(self.device)))

    def training_step(self, batch, batch_idx):
        velocity = self.model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
        temporal_weight = self.mlp(torch.tensor([self.subject_t1['age']]).to(self.device))
        forward_flow, backward_flow = self.model.velocity_to_flow(velocity=velocity * temporal_weight)
        loss_tensor = self.model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            subject = self.trainer.train_dataloader.dataset.dataset[i + 1]
            temporal_weight = self.mlp(torch.tensor([subject['age']]).to(self.device))
            forward_flow, backward_flow = self.model.velocity_to_flow(velocity=velocity * temporal_weight)
            loss_tensor += self.model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)

        loss_zero = F.mse_loss(self.mlp(torch.tensor([0.0]).to(self.device)), torch.tensor([0.0]).to(self.device))
        loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float() + loss_zero
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("MLP t_0", loss_zero, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.model.eval()
        self.mlp.eval()
        if self.current_epoch % 10 == 0:
            self.model.model.eval()
            with torch.no_grad():
                velocity = self.model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
                temporal_weight = self.mlp(torch.tensor([1.0]).to(self.device))
                forward_flow, backward_flow = self.model.velocity_to_flow(velocity=velocity * temporal_weight)
                label_warped_source = self.model.warp(self.subject_t0['label'][tio.DATA].to(self.device).float(),
                                                      forward_flow)
                image_warped_source = self.model.warp(self.subject_t0['image'][tio.DATA].to(self.device).float(),
                                                      forward_flow)
                tio.LabelMap(tensor=torch.argmax(label_warped_source, dim=1).int().detach().cpu().numpy(),
                             affine=self.subject_t0['label'].affine).save(self.save_path +  "/label_warped_source.nii.gz")
                tio.ScalarImage(tensor=image_warped_source.squeeze(0).detach().cpu().numpy(),
                                affine=self.subject_t0['image'].affine).save(self.save_path + "/image_warped_source.nii.gz")
                for subject in self.trainer.train_dataloader.dataset.dataset:
                    temporal_weight = self.mlp(torch.tensor([subject['age']]).to(self.device))
                    forward_flow, backward_flow = self.model.velocity_to_flow(velocity * temporal_weight)
                    warped_source_label = self.model.warp(self.subject_t0['label'][tio.DATA].to(self.device).float(), forward_flow)
                    self.dice_metric(torch.round(warped_source_label), subject['label'][tio.DATA].to(self.device).float().unsqueeze(0))
                overall_dice = self.dice_metric.aggregate()
                self.dice_metric.reset()
                mean_dices = torch.mean(overall_dice).item()
            if self.dice_max < mean_dices:
                self.dice_max = mean_dices
                print("New best dice:", self.dice_max)
                torch.save(self.model.state_dict(), self.save_path + "/model_linear_best.pth")
                torch.save(self.mlp.state_dict(), self.save_path + "/mlp_model_best.pth")
            x = np.arange(0, 1, 0.01)
            y = []
            x_label = []
            for i in x:
                temporal_weight = self.mlp(torch.asarray([i]).float().to(self.device))
                y.append(temporal_weight.detach().cpu().numpy())
                x_label.append(i)
            plt.figure(figsize=(10, 8))
            plt.plot(x, y)
            self.trainer.loggers[0].experiment.add_figure("Temporal MLP coefficient", plt.gcf(), global_step=0)
            plt.close()
        torch.save(self.model.state_dict(), self.save_path + "/last_reg_model.pth")
        torch.save(self.mlp.state_dict(), self.save_path + "/last_mlp_model.pth")





class TemporalTrajectoryActiveLearningMLP(TemporalTrajectoryMLPSVF):
    def __init__(self, reg_model: RegistrationModuleSVF, mlp_model : MLP, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 2, save_path: str = ""):
        super().__init__(reg_model, mlp_model, loss, lambda_sim, lambda_seg, lambda_mag, lambda_grad, num_classes, save_path)
        self.subject_t0 = None
        self.subject_t1 = None
        self.index_train_current_epoch = None
        self.num_inter_by_epoch = 6

    def on_train_epoch_start(self) -> None:
        self.model.train()
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
        velocity = self.model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
        for i in self.index_train_current_epoch:
            self.counter_samples_indexes[i] += 1
            subject = self.trainer.train_dataloader.dataset.dataset[i]
            coef = self.temporal_mlp(torch.asarray([subject['age']]).to(self.device))
            forward_flow, backward_flow = self.model.velocity_to_flow(velocity * coef)
            loss_intermediate_tensor += self.model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        losses = loss_intermediate_tensor
        loss_zero = F.mse_loss(self.temporal_mlp(torch.tensor([0.0]).to(self.device)), torch.tensor([0.0]).to(self.device)) # Lock the deformation to 0 at t=0
        loss = torch.sum(losses)
        loss = loss + 1.0 * loss_zero
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", losses[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", losses[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", losses[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", losses[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("T0_loss_mlp", loss_zero, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.model.eval()
        self.temporal_mlp.eval()
        all_dice = []
        accumulate_errors = torch.tensor([]).to(self.device)
        with torch.no_grad():
            velocity = self.model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
            for subject in self.trainer.train_dataloader.dataset.dataset:
                temporal_weight = self.temporal_mlp(torch.tensor([subject['age']]).to(self.device))
                forward_flow, backward_flow = self.model.velocity_to_flow(velocity * temporal_weight)
                loss = self.model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow,
                                                        lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg,
                                                        lambda_mag=0, lambda_grad=0)
                accumulate_errors = torch.cat((accumulate_errors, torch.tensor([torch.sum(loss)]).to(self.device)))
                warped_source_label = self.model.warp(self.subject_t0['label'][tio.DATA].to(self.device).float(),
                                                          forward_flow)
                dice = dice_score(warped_source_label.detach().squeeze().cpu().numpy(),
                                  subject['label'][tio.DATA].squeeze().long().detach().cpu().numpy())
                current_dice = np.mean(dice)
                mean_dice = np.mean(dice)
                all_dice.append(mean_dice)
            cumulative_loss = torch.cumsum(accumulate_errors, dim=0)
            cumulative_loss_normalized = (cumulative_loss - torch.min(cumulative_loss)) / (
                    torch.max(cumulative_loss) - torch.min(cumulative_loss))
            self.index_train_current_epoch = [torch.abs(cumulative_loss_normalized - random.random()).argmin().cpu().numpy()
                                              for _ in range(self.num_inter_by_epoch)]
            mean_all_dice = np.mean(all_dice)
            self.log("Mean dice", mean_all_dice)
            if mean_all_dice > self.dice_max:
                self.dice_max = mean_all_dice
                print("New best dice:", self.dice_max)
                torch.save(self.reg_model.state_dict(), self.save_path + "/reg_model_best.pth")
                torch.save(self.temporal_mlp.state_dict(), self.save_path + "/mlp_model_best.pth")

        torch.save(self.reg_model.state_dict(), self.save_path + "/last_reg_model.pth")
        torch.save(self.temporal_mlp.state_dict(), self.save_path + "/last_mlp_model.pth")
