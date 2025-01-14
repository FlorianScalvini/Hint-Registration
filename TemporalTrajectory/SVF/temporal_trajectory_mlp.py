import random
import torchio as tio
from network import MLP
from torch import Tensor
from utils.loss import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Registration import RegistrationModuleSVF
from temporal_trajectory import TemporalTrajectorySVF

class TemporalTrajectoryMLPSVF(TemporalTrajectorySVF):
    '''
        Temporal Trajectory module using MLP for temporal weighting
    '''
    def __init__(self, reg_model: RegistrationModuleSVF, mlp_model : MLP, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 20, num_inter_by_epoch=4, save_path: str = ""):
        super().__init__(reg_model, loss, lambda_sim, lambda_seg, lambda_mag, lambda_grad, save_path, num_classes, num_inter_by_epoch)
        self.mlp = mlp_model


    def forward(self, source: Tensor, target: Tensor, age: Tensor):
        velocity = self.reg_model(source, target)
        coef = torch.abs(self.mlp(age))
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
        return forward_flow, backward_flow

    def on_train_epoch_start(self) -> None:
        self.reg_model.train()
        self.mlp.train()


    def on_validation_epoch_start(self):
        self.reg_model.eval()
        self.mlp.eval()


    def on_train_start(self) -> None:
        self.counter_samples_indexes = np.zeros(len(self.trainer.train_dataloader.dataset.dataset)) # Counter for the number of times a sample has been used
        # Get the t0 and t1 subjects
        for i in range(len(self.trainer.train_dataloader.dataset.dataset)):
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 0:
                self.subject_t0 = self.trainer.train_dataloader.dataset.dataset[i]
            if self.trainer.train_dataloader.dataset.dataset[i]['age'] == 1:
                self.subject_t1 = self.trainer.train_dataloader.dataset.dataset[i]
        self.subject_t0['image'][tio.DATA] = self.subject_t0['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t0['label'][tio.DATA] = self.subject_t0['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['image'][tio.DATA] = self.subject_t1['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device)
        self.subject_t1['label'][tio.DATA] = self.subject_t1['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)


    def training_step(self, batch, batch_idx):
        # Compute the registration between T0 and T1
        velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA]) # Compute the velocity field
        temporal_weight = torch.abs(self.mlp(torch.tensor([self.subject_t1['age']]).to(self.device))) # Compute the temporal weight
        forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity * temporal_weight) # Compute the flow
        loss_tensor = self.reg_model.registration_loss(self.subject_t0, self.subject_t1, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad) # Compute the loss between T0 and T1
        index = random.sample(range(0, len(self.trainer.train_dataloader.dataset.dataset) - 2), self.num_inter_by_epoch)
        for i in index:
            subject = self.trainer.train_dataloader.dataset.dataset[i + 1]
            temporal_weight = torch.abs(self.mlp(torch.tensor([subject['age']]).to(self.device)))
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity * temporal_weight)
            loss_tensor += self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
        # Compute the global loss
        loss_zero = F.mse_loss(torch.abs(self.mlp(torch.tensor([0.0]).to(self.device))), torch.tensor([0.0]).to(self.device))
        loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float() + loss_zero
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("MLP t_0", loss_zero, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
            weighted_age = torch.abs(self.mlp(torch.tensor([subject_t1['age']]).to(self.device)))
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity * weighted_age)
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
            torch.save(self.reg_model.state_dict(), self.save_path + "/reg_model.pth")
            torch.save(self.mlp.state_dict(), self.save_path + "/mlp_best.pth")

        x = np.arange(0, 1, 0.01)
        y = []
        x_label = []
        for i in x:
            temporal_weight = torch.abs(self.mlp(torch.asarray([i]).float().to(self.device)))
            y.append(temporal_weight.detach().cpu().numpy())
            x_label.append(i)
        plt.figure(figsize=(10, 8))
        plt.plot(x, y)
        self.trainer.loggers[0].experiment.add_figure("Temporal MLP coefficient", plt.gcf(), global_step=0)
        plt.close()



    def on_train_epoch_end(self):
        # Set the model in evaluation mode
        # Save the model at the end of the epoch
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_reg_model.pth")
        torch.save(self.mlp.state_dict(), self.save_path + "/last_mlp_model.pth")

    def load_model(self, path: str):
        self.reg_model.load_state_dict(torch.load(path))

    def load_mlp_model(self, mlp_path: str):
        self.mlp.load_state_dict(torch.load(mlp_path))

class TemporalTrajectoryActiveLearningMLP(TemporalTrajectoryMLPSVF):
    def __init__(self, reg_model: RegistrationModuleSVF, mlp_model : MLP, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 2, num_inter_by_epoch=1, save_path: str = ""):
        super().__init__(reg_model, mlp_model, loss, lambda_sim, lambda_seg, lambda_mag, lambda_grad, num_classes, num_inter_by_epoch, save_path)
        self.index_train_current_epoch = None

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
        for i in self.index_train_current_epoch:
            self.counter_samples_indexes[i] += 1
            subject = self.trainer.train_dataloader.dataset.dataset[i]
            coef = self.temporal_mlp(torch.asarray([subject['age']]).to(self.device))
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * coef)
            loss_intermediate_tensor += self.reg_model.registration_loss(self.subject_t0, subject, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad)
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
        self.reg_model.eval()
        self.temporal_mlp.eval()
        all_dice = []
        accumulate_errors = torch.tensor([]).to(self.device)
        with torch.no_grad():
            # Compute the velocity field between T0 and T1
            velocity = self.reg_model(self.subject_t0['image'][tio.DATA], self.subject_t1['image'][tio.DATA])
            temporal_weight = torch.abs(self.mlp(torch.tensor([1.0]).to(self.device)))
            forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity=velocity * temporal_weight)
            label_warped_source = self.reg_model.warp(self.subject_t0['label'][tio.DATA].to(self.device).float(), forward_flow)
            image_warped_source = self.reg_model.warp(self.subject_t0['image'][tio.DATA].to(self.device).float(), forward_flow)
            tio.LabelMap(tensor=torch.argmax(label_warped_source, dim=1).int().detach().cpu().numpy(),
                         affine=self.subject_t0['label'].affine).save(self.save_path +  "/label_warped_source.nii.gz")
            tio.ScalarImage(tensor=image_warped_source.squeeze(0).detach().cpu().numpy(),
                            affine=self.subject_t0['image'].affine).save(self.save_path + "/image_warped_source.nii.gz")

            # Compute the DICE Score for each subject
            for subject in self.trainer.train_dataloader.dataset.dataset:
                temporal_weight = torch.abs(self.mlp(torch.tensor([subject['age']]).to(self.device)))
                forward_flow, backward_flow = self.reg_model.velocity_to_flow(velocity * temporal_weight)
                warped_source_label = self.reg_model.warp(self.subject_t0['label'][tio.DATA].to(self.device).float(), forward_flow)
                self.dice_metric(torch.round(warped_source_label), subject['label'][tio.DATA].to(self.device).float().unsqueeze(0))

            # Compute the mean DICE Score
            overall_dice = self.dice_metric.aggregate() # Compute the DICE Score
            self.dice_metric.reset() # Reset the dice metric
            mean_dices = torch.mean(overall_dice).item()
            all_dice.append(mean_dices)
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
