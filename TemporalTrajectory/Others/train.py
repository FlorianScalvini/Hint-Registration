import torch
import torchio as tio
from Registration import RegistrationModuleSVF
from TemporalTrajectory.temporal_trajectory import TemporalTrajectorySVF

class LongitudinalAtlasMeanVelocityModule(TemporalTrajectorySVF):
    def __init__(self, regnet: RegistrationModuleSVF, loss: str = 'mse', lambda_sim: float = 1.0,
                 lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__(reg_model=regnet, loss=loss, lambda_sim=lambda_sim, lambda_seg=lambda_seg, lambda_mag=lambda_mag, lambda_grad=lambda_grad)
        self.mDice = 0
        self.weightedMeanVelocity = None

    def on_train_epoch_end(self):
        torch.save(self.regnet.state_dict(), "./model_last_epoch.pth")
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
            forward_flow, backward_flow = self.regnet.velocity_to_flow(weightedMeanVelocity * weight)
            loss_tensor += self.regnet.registration_loss(batch, target_subject, forward_flow, backward_flow)

        loss = self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", self.lambda_sim * loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss



