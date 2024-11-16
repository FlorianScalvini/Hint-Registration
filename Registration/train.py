import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import monai
import torchio as tio
from dataset import PairwiseSubjectsDataset
import argparse
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils.loss import Grad3d
from Registration import RegistrationModuleSVF, RegistrationModule


class RegistrationTrainingModule(pl.LightningModule):
    def __init__(self, model : RegistrationModuleSVF | RegistrationModule, lambda_sim: float = 1.0, lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__()
        self.model = model
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        return self.model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch):
        source, target = batch.values()
        forward_flow, backward_flow = self.model.forward_backward_flow_registration(source['image'][tio.DATA], target['image'][tio.DATA])
        loss_tensor = self.model.registration_loss(source, target, forward_flow, backward_flow)

        loss = (loss_tensor[0] * self.lambda_sim + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", self.lambda_sim * loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


def main(arguments):

    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20)
    ])

    train_dataset = PairwiseSubjectsDataset(dataset_path=arguments.csv_path, transform=train_transform, age=False)
    training_loader = tio.SubjectsLoader(train_dataset, batch_size=1, num_workers=8, persistent_workers=True)

    ## Config training
    trainer_args = {
        'max_epochs': arguments.epochs,
        'strategy': DDPStrategy(find_unused_parameters=True),
        'precision': arguments.precision,
        'accumulate_grad_batches': arguments.accumulate_grad_batches,
    }

    if arguments.logger is None:
        trainer_args['logger'] = False
        trainer_args['enable_checkpointing'] = False
    else:
        trainer_args['logger'] = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=arguments.logger)

    model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=[128, 128, 128], int_steps=7)
    #model = RegistrationModule(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=[128, 128, 128])
    reg_net = RegistrationTrainingModule(model=model, lambda_sim=arguments.lam_l, lambda_seg=arguments.lam_s, lambda_mag=arguments.lam_m, lambda_grad=arguments.lam_g)
    trainer_reg = pl.Trainer(**trainer_args)
    # %%
    trainer_reg.fit(reg_net, train_dataloaders=training_loader, val_dataloaders=None)
    torch.save(reg_net.model.state_dict(), arguments.save)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration Train 3D Images')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False, default="../data/full_dataset.csv")
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default=500)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required=False,
                        default=1)
    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append',
                        required=False, default="mse")
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, required=False, default=1)
    parser.add_argument('--lam_s', help='Lambda loss for image segmentation', type=float, required=False, default=1)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required=False, default=0.01)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required=False, default=0.05)
    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./output")
    parser.add_argument('--load', help='Input model', type=str, required=False, default=None)
    parser.add_argument('--save', help='Output model', type=str, required=False, default="./model_attention_unet_svf.pth")
    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required=False,
                        default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)
    main(parser.parse_args())
    print("Success!")