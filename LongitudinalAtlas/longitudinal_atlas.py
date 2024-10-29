import monai.networks.nets
import torchio as tio
import pytorch_lightning as pl
from utils.loss import *
import torch.nn.functional as F
from utils.metric import dice_score
from net.network import RegistrationModelSVF
from torch import Tensor


class LongitudinalAtlas(pl.LightningModule):
    def __init__(self, shape: [int, int, int], int_steps: int=7, loss:list=['mse'], lambda_loss:list=[1], lambda_seg: float=0, lambda_mag:float=0, lambda_grad: float=0):
        super().__init__()
        self.model = RegistrationModelSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)),
                                          inshape=shape,
                                          int_steps=int_steps)

        self.loss = []
        for l in loss:
            self.loss.append(GetLoss(l))
        self.lambda_seg = lambda_seg
        self.lambda_loss = lambda_loss
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.mDice = 0
    def forward(self, source: Tensor, target: Tensor, time: float):
        return self.model(source, target, time)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def getLoss(self, source: Tensor, target: Tensor, forward_flow: Tensor, backward_flow: Tensor):
        loss_pair = torch.zeros((4)).to(self.device).float()
        loss_pair[0] = self.loss[0](target['image'][tio.DATA], self.model.warped_image(source, forward_flow)) + \
                       self.loss[0](source['image'][tio.DATA], self.model.warped_image(target, backward_flow))

        if self.lambda_seg > 0:
            warped_source_label = self.model.warped_image(source['label'][tio.DATA].float(), forward_flow)
            warped_target_label = self.model.warped_image(target['label'][tio.DATA].float(), backward_flow)
            loss_pair[1] += F.mse_loss(target['label'][tio.DATA][:, 1::, ...].float(), warped_source_label[:, 1::, ...]) \
                            + F.mse_loss(source['label'][tio.DATA][:, 1::, ...].float(), warped_target_label[:, 1::, ...])

        if self.lambda_mag > 0:
            loss_pair[2] += F.mse_loss(torch.zeros(forward_flow.shape, device=self.device), forward_flow) + \
                            F.mse_loss(torch.zeros(backward_flow.shape, device=self.device), backward_flow)

        if self.lambda_grad > 0:
            loss_pair[3] += Grad3d().forward(forward_flow) + Grad3d().forward(backward_flow)
        return loss_pair


    def training_step(self, batch, batch_idx):
        sample1, sample2, sample3 = batch.values()

        forward_flow, backward_flow = self(sample1['image'][tio.DATA].float(), sample2['image'][tio.DATA].float(), sample2['age'][0] - sample1['age'][0])
        loss_tensor = self.getLoss(sample1, sample2, forward_flow, backward_flow)

        forward_flow, backward_flow = self(sample1['image'][tio.DATA].float(), sample3['image'][tio.DATA].float(), sample3['age'][0] - sample1['age'][0])
        loss_tensor += self.getLoss(sample1, sample3, forward_flow, backward_flow)

        forward_flow, backward_flow = self(sample2['image'][tio.DATA].float(), sample3['image'][tio.DATA].float(), sample3['age'][0] - sample2['age'][0])
        loss_tensor += self.getLoss(sample2, sample3, forward_flow, backward_flow)

        loss = (loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self) -> None:
        self.mDice = 0

    def validation_step(self, batch) :
        sample1, sample2, _ = batch.values()
        forward_flow, backward_flow = self(sample1['image'][tio.DATA].float(), sample2['image'][tio.DATA].float(), sample2['age'][0] - sample1['age'][0])
        warped_source_label = self.model.warped_image(sample1['label'][tio.DATA].float(), forward_flow)
        warped_warped_source_label_argmax = torch.argmax(warped_source_label, dim=1)
        warped_target_argmax = torch.argmax(sample2['label'][tio.DATA].float(), dim=1)
        dice_stt = dice_score(warped_warped_source_label_argmax, warped_target_argmax, num_classes=20)
        self.mDice += np.mean(dice_stt)

    def on_validation_end(self):
        self.logger.experiment.add_scalar("Dice", self.mDice / len(self.trainer.val_dataloaders), self.current_epoch)