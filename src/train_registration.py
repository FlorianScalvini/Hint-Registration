import os
import torch
import monai
import argparse
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from pytorch_lightning.callbacks import ModelCheckpoint
from losses import Grad3d, GetLoss, InverseConsistency
from modules.data import RegistrationDataModule
from modules.registration import RegistrationModuleSVF, RegistrationModule
from utils import create_directory, write_namespace_arguments
import torchio as tio


class RegistrationTrainingModule(pl.LightningModule):
    """
    Registration training module for 3D image registration
    """
    def __init__(self, model : RegistrationModuleSVF | RegistrationModule, learning_rate: float= 0.001,
                 similarity_loss: str ='mi', lambda_sim: float = 1.0, lambda_seg: float = 0, lambda_mag: float = 0,
                 lambda_grad: float = 0, lambda_inv: float = 0, save_path: str = "./", penalize: str = 'v'):
        """
        Registration training module for 3D image registration
        :param model: RegistrationModuleSVF or RegistrationModule
        :param learning_rate: Learning rate for the optimizer
        :param similarity_loss: Similarity loss function (mi, mse, ncc)
        :param lambda_sim: Weight for the similarity loss
        :param lambda_seg: Weight for the segmentation loss
        :param lambda_mag: Weight for the magnitude loss
        :param lambda_grad: Weight for the gradient loss
        :param lambda_inv: Weight for the inverse consistency loss
        :param save_path: Path to save the model
        :param penalize: Regularization of the velocity or displacement field
        """
        super().__init__()
        self.model = model
        self.sim_loss = GetLoss(loss=similarity_loss)
        self.grad_loss = Grad3d(penalty='l2')
        self.mag_loss = nn.MSELoss()
        self.seg_loss = nn.MSELoss()
        self.inv_loss = InverseConsistency()
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.lambda_inv = lambda_inv
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
        return self.model(source, target)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.dice_max = 0
        self.model.train()

    def training_step(self, batch) -> torch.Tensor:
        """
        Training step for the registration module
        :param batch: Pair of subjects (Source, Target)
        :return: Loss value
        """
        source, target = batch.values()
        velocity = self.forward(source['image'][tio.DATA], target['image'][tio.DATA])
        back_velocity = -velocity
        forward_flow, backward_flow = self.model.velocity2displacement(velocity), self.model.velocity2displacement(back_velocity)
        losses = (self.registration_loss(source, target, forward_flow, backward_flow, velocity)
                  + self.registration_loss(target, source, backward_flow, forward_flow, back_velocity))
        loss = torch.sum(losses)
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", losses[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", losses[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", losses[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", losses[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Inverse consistency", losses[4], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), self.save_path + "/last_model.pth")

    def validation_step(self, batch):
        self.dice_metric.reset()
        source, target = batch.values()
        dice_scores = []
        with torch.no_grad():
            velocity = self.forward(source['image'][tio.DATA], target['image'][tio.DATA])
            forward_flow, backward_flow = self.model.velocity2displacement(velocity), self.model.velocity2displacement(-velocity)
        warped_source_label = self.model.warp(source['label'][tio.DATA].float().to(self.device), forward_flow)
        warped_source_label = tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu().numpy(), affine=target['label']['affine'][0])
        warped_source_label = tio.OneHot(args.num_classes)(warped_source_label)
        dice = self.dice_metric(warped_source_label[tio.DATA].to(self.device).unsqueeze(0), target['label'][tio.DATA].float().to(self.device))[0]
        dice_scores.append(torch.mean(dice[1:]).cpu().numpy())
        losses = self.registration_loss(source, target, forward_flow, backward_flow, )
        loss = torch.sum(losses)
        mean_dices =  sum(dice_scores) / len(dice_scores)
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.model.state_dict(), self.save_path + "/best_model.pth")
        self.log("Mean dice", mean_dices, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", losses[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", losses[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", losses[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", losses[3], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Inverse consistency", losses[4], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss



    def registration_loss(self,
                          source: tio.Subject,
                          target: tio.Subject,
                          f: Tensor,
                          bf: Tensor,
                          v: Tensor = None) -> Tensor:
        '''
        Compute the registration.yaml loss for a pair of subjects.
        :param source: Source subject
        :param target: Target subject
        :param f: Flow field
        :param bf: Backward flow field
        :param v: Velocity field
        :return: Loss value
        '''
        # Initialize loss tensor
        loss_pair = torch.zeros(5).float().to(self.device)

        # Get and preprocess images
        target_image = target['image'][tio.DATA].to(self.device)
        source_image = source['image'][tio.DATA].to(self.device)

        # Similarity loss
        if self.lambda_sim > 0:
            loss_pair[0] = self.lambda_sim * self.sim_loss(self.model.warp(source_image, f), target_image)

        # Segmentation loss
        if self.lambda_seg > 0:
            target_label = target['label'][tio.DATA].float().to(self.device)
            source_label = source['label'][tio.DATA].float().to(self.device)
            warped_source_label = self.model.warp(source_label, f)
            loss_pair[1] = self.lambda_seg * self.seg_loss(warped_source_label[:, 1:, ...], target_label[:, 1:, ...])

        # Magnitude loss
        if self.lambda_mag > 0:
            loss_pair[2] = self.lambda_mag * self.mag_loss(v, torch.zeros_like(v)) \
                if self.penalize == 'v' else self.mag_loss(f, torch.zeros_like(f))

        # Gradient loss
        if self.lambda_grad > 0:
            loss_pair[3] = self.lambda_grad * self.grad_loss(v) if self.penalize == 'v' else self.mag_loss(f)

        # Inverse consistency loss
        if self.lambda_inv > 0:
            loss_pair[4] = self.lambda_inv * self.inv_loss(f, bf)
        return loss_pair



def train(args):
    try:
        reg_model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(dropout=0.1, spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32, 64], strides=[2,2,2]), int_steps=7)
        if args.load != "":
            reg_model.load_state_dict(torch.load(args.load))
    except:
        raise ValueError("Model initialization failed")

    trainer_args = {'max_steps': args.max_steps, 'precision': args.precision,
                    'accumulate_grad_batches': args.accumulate_grad_batches,
                    'logger': pl.loggers.TensorBoardLogger(save_dir="./regis/log", name=None),
                    'check_val_every_n_epoch': 20, 'num_sanity_val_steps': 0,
                    'callbacks': [ ModelCheckpoint(every_n_train_steps=100)]
                    }

    save_path = trainer_args['logger'].log_dir.replace("log", "Results")
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "configs.json"))

    # Train the model
    model = RegistrationTrainingModule(
        model=reg_model,
        lambda_sim=args.lam_l,
        lambda_seg=args.lam_s,
        lambda_mag=args.lam_m,
        lambda_grad=args.lam_g,
        lambda_inv=args.lam_i,
        learning_rate=args.lr,
        penalize=args.penalize,
        save_path=save_path
    )

    trainer = pl.Trainer(**trainer_args)
    checkpoint = None
    if args.checkpoint != "":
        checkpoint = args.checkpoint
    trainer.fit(model=model,
                datamodule=RegistrationDataModule(args.csv, batch_size=1, rsize=args.rsize, csize=args.csize),
                ckpt_path=checkpoint)
    torch.save(model.model.state_dict(), os.path.join(save_path, "final_model_reg.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Registration 3D Images')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='C:/Users/Florian/Documents/GitHub/Hint-Registration/data/full_dataset.csv')
    parser.add_argument('--max_steps', type=int, help='Number of steps', default=15000)
    parser.add_argument('--accumulate_grad_batches', type=int, help='Number of batches to accumulate', default=1)
    parser.add_argument('--loss', type=str, help='Loss function', default='mi')
    parser.add_argument('--lam_l', type=float, help='Lambda similarity weight', default=4)
    parser.add_argument('--lam_s', type=float, help='Lambda segmentation weight', default=100)
    parser.add_argument('--lam_m', type=float, help='Lambda magnitude weight', default=0.1)
    parser.add_argument('--lam_g', type=float, help='Lambda gradient weight', default=0.00004)
    parser.add_argument('--lam_i', type=float, help='Lambda gradient weight', default=0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--precision', type=int, help='Precision', default=32)
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize shape', default=[160, 160, 160])
    parser.add_argument('--csize', type=int, nargs='+', help='Cropsize shape', default=[221, 221, 221])
    parser.add_argument('--tensor-cores', type=bool, help='Use tensor cores', default=False)
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--load', type=str, help='Load registration.yaml model', default="")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint', default="")
    parser.add_argument('--penalize', type=str, help='Regularization of the velocity or displacement field', default='v')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    train(args=args)