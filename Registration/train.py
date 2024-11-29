import sys
sys.path.insert(0, "/home/florian/Documents/Programs/Hint-Registration")

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import monai
import torchio as tio
from dataset import PairwiseSubjectsDataset
import argparse
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils import dice_score, create_directory
from Registration import RegistrationModuleSVF, RegistrationModule


class RegistrationTrainingModule(pl.LightningModule):
    def __init__(self, model : RegistrationModuleSVF | RegistrationModule, lambda_sim: float = 1.0, lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, num_classes: int = 20, save_path: str = "./"):
        super().__init__()
        self.model = model
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.num_classes = num_classes
        self.save_path = save_path

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        return self.model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.model.train()
        self.dice_max = 0
    def training_step(self, batch):
        source, target = batch.values()
        forward_flow, backward_flow = self.model.forward_backward_flow_registration(source['image'][tio.DATA], target['image'][tio.DATA])
        loss_tensor = self.model.registration_loss(source, target, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad, device=self.device, num_classes=self.num_classes)

        loss = loss_tensor.sum()

        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude",  loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude",  loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient",  loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def on_train_epoch_end(self):
        self.model.eval()
        dice_all = []
        if self.current_epoch % 10 == 0:
            for i in self.trainer.train_dataloader.dataset:
                source, target = i.values()
                forward_flow, backward_flow = self.model.forward_backward_flow_registration(source['image'][tio.DATA].unsqueeze(dim=0).to(self.device), target['image'][tio.DATA].unsqueeze(dim=0).to(self.device))
                source_label_warped = self.model.warp(source['label'][tio.DATA].unsqueeze(dim=0).to(self.device).float(), forward_flow, mode='nearest')
                one_hot_warped_source = F.one_hot(source_label_warped.squeeze().long(), num_classes=self.num_classes).float().permute(3, 0, 1, 2)
                one_hot_target = F.one_hot(target['label'][tio.DATA].squeeze().long(), num_classes=self.num_classes).permute(3, 0, 1, 2)
                dice = dice_score(one_hot_warped_source.detach().cpu().numpy(), one_hot_target.detach().cpu().numpy())
                dice_all.append(np.mean(dice[1::]))
            mean_dice_all = np.mean(dice_all)
            self.log("Mean dice", mean_dice_all, prog_bar=True, on_epoch=True, sync_dist=True)
            if self.dice_max < mean_dice_all:
                self.dice_max = mean_dice_all
                torch.save(self.model.state_dict(), self.save_path + "/best_model.pth")
            torch.save(self.model.state_dict(), self.save_path + "/last_model.pth")

def main(arguments):

    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128)
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
        trainer_args['logger'] = pl.loggers.TensorBoardLogger(save_dir= "./" + arguments.logger, name=None)
    model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=[128, 128, 128], int_steps=7)
    #model = RegistrationModule(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=[128, 128, 128])


    save_path = trainer_args['logger'].log_dir.replace(arguments.logger, "Results")
    create_directory(save_path)

    reg_net = RegistrationTrainingModule(model=model, lambda_sim=arguments.lam_l, lambda_seg=arguments.lam_s, lambda_mag=arguments.lam_m, lambda_grad=arguments.lam_g, num_classes=arguments.num_classes, save_path=save_path)
    trainer_reg = pl.Trainer(**trainer_args)
    # %%
    trainer_reg.fit(reg_net, train_dataloaders=training_loader, val_dataloaders=None)
    torch.save(reg_net.model.state_dict(), arguments.save)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration Train 3D Images')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False, default="../data/full_dataset.csv")
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default=200)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required=False,
                        default=1)
    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append',
                        required=False, default="mse")
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, required=False, default=1)
    parser.add_argument('--lam_s', help='Lambda loss for image segmentation', type=float, required=False, default=10)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required=False, default=0.001)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required=False, default=0.001)
    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./output")
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./best_model.pth")
    parser.add_argument('--save', help='Output model', type=str, required=False, default="./model_attention_unet_svf_2.pth")
    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required=False,
                        default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_classes', help='Number of classes', type=int, required=False, default=20)
    main(parser.parse_args())
    print("Success!")