import os
import torch
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from losses import PairwiseRegistrationLoss
from modules.pairwise_registration import PairwiseRegistrationModuleVelocity
from losses import NCCLoss

class RegistrationTrainingModule(pl.LightningModule):
    """
    Registration training module for 3D image registration
    """
    def __init__(self, model : PairwiseRegistrationModuleVelocity, loss: PairwiseRegistrationLoss, learning_rate: float= 0.001,
                 save_path: str = "./", penalize: str = 'v'):
        """
        Registration training module for 3D image registration
        :param model: RegistrationModuleSVF
        :param learning_rate: Learning rate for the optimizer
        :param loss: PairwiseRegistrationLoss
        :param save_path: Path to save the model
        :param penalize: Regularization of the velocity or displacement field
        """
        super().__init__()
        self.reg_model = model
        self.loss = loss
        if penalize not in ['v', 'd']:
            raise ValueError("Penalize must be 'v' or 'd'")
        self.penalize = penalize
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.dice_metric = DiceMetric(include_background=True, reduction="none")
        self.dice_max = 0

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the registration module
        :param source: Source image
        :param target: Target image
        :return: Flow field if RegistrationModule else
        """
        return self.reg_model(source, target)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.dice_max = 0
        self.reg_model.train()

    def training_step(self, batch) -> torch.Tensor:
        """
        Training step for the registration module
        :param batch: Pair of subjects (Source, Target)
        :return: Loss value
        """
        source, target = batch.values()
        velocity = self.forward(source['image'][tio.DATA].float(), target['image'][tio.DATA].float())
        back_velocity = -velocity
        forward_flow, backward_flow = self.reg_model.velocity2displacement(velocity), self.reg_model.velocity2displacement(back_velocity)
        warped_source = self.reg_model.warp(source['image'][tio.DATA], forward_flow)
        warped_source_label = self.reg_model.warp(source['label'][tio.DATA].float(), forward_flow)
        losses = self.loss(warped_source, target['image'][tio.DATA], warped_source_label.float(),
                           target['label'][tio.DATA].float(), forward_flow, backward_flow,
                           None if self.penalize == 'd' else velocity)
        loss = torch.sum(losses)
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", losses[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", losses[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", losses[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", losses[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Inverse consistency", losses[4], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
        """

        t = torch.rand(1, device=velocity.device)
        flow_J = self.reg_model.velocity2displacement(t * velocity)
        Jw = self.reg_model.warp(source['label'][tio.DATA].float(), flow_J)

        flow_I = self.reg_model.velocity2displacement((t - 1.) * velocity)
        Iw = self.reg_model.warp(target['label'][tio.DATA].float(), flow_I)

        image_loss = torch.nn.MSELoss()(Jw, Iw)


        flow_2t_1 = self.reg_model.velocity2displacement(((2. * t) - 1.) * velocity)
        flow_JI =  self.reg_model.warp(flow_J, flow_I)  + flow_I
        flow_IJ = self.reg_model.warp(flow_I, flow_J) + flow_J

        flow_loss = 0.5 * (torch.mean((flow_2t_1 - flow_JI) ** 2) + torch.mean((flow_2t_1 - flow_IJ) ** 2))


        loss = image_loss + flow_loss
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", image_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Regu", flow_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

        """


    def on_train_epoch_end(self):
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_model.pth")

    def validation_step(self, batch):
        self.dice_metric.reset()
        source, target = batch.values()
        dice_scores = []
        with torch.no_grad():
            velocity = self.forward(source['image'][tio.DATA], target['image'][tio.DATA])
            forward_flow = self.reg_model.velocity2displacement(velocity)
            backward_flow = self.reg_model.velocity2displacement(-velocity)
        warped_source = self.reg_model.warp(source['image'][tio.DATA], forward_flow)
        warped_source_label = self.reg_model.warp(source['label'][tio.DATA].float(), forward_flow)
        ## Compute the loss
        losses = self.loss(warped_source, target['image'][tio.DATA],
                           warped_source_label.float(), target['label'][tio.DATA].float(),
                           forward_flow, backward_flow, None if self.penalize == 'd' else velocity)
        loss = torch.sum(losses)

        ## Compute the DICE Score
        warped_source_label = tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu().numpy(), affine=target['label']['affine'][0])
        warped_source_label = tio.OneHot(source['label'][tio.DATA].shape[1])(warped_source_label)
        dice = self.dice_metric(warped_source_label[tio.DATA].to(self.device).unsqueeze(0), target['label'][tio.DATA].float().to(self.device))[0]
        dice_scores.append(torch.mean(dice[1:]).cpu().numpy())
        mean_dices =  sum(dice_scores) / len(dice_scores)
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.reg_model.state_dict(), self.save_path + "/best_model.pth")
        self.log("Mean dice", mean_dices, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", losses[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", losses[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", losses[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", losses[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Inverse consistency", losses[4], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def save(self, path: str):
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.reg_model.state_dict(), os.path.join(path, "model.pth"))
