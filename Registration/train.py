import sys
sys.path.insert(0, "/home/florian/Documents/Programs/Hint-Registration")
import os
import json
import torch
import argparse
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from dataset import PairwiseSubjectsDataset
from pytorch_lightning.strategies.ddp import DDPStrategy
from Registration import RegistrationModuleSVF, RegistrationModule
from utils import get_model_from_string, create_directory, write_text_to_file, config_dict_to_markdown


class RegistrationTrainingModule(pl.LightningModule):
    def __init__(self, model : RegistrationModuleSVF | RegistrationModule, lambda_sim: float = 1.0, lambda_seg: float = 0, lambda_mag: float = 0, lambda_grad: float = 0, save_path: str = "./"):
        super().__init__()
        self.model = model
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad
        self.save_path = save_path
        self.dice_metric = DiceMetric(include_background=True, reduction="none")

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        return self.model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.dice_max = 0
        self.model.train()

    def training_step(self, batch):
        '''
            Compute the loss of the model on each pair of images
        '''
        source, target = batch.values()
        forward_flow, backward_flow = self.model.forward_backward_flow_registration(source['image'][tio.DATA], target['image'][tio.DATA])
        loss_tensor = self.model.registration_loss(source, target, forward_flow, backward_flow, lambda_sim=self.lambda_sim, lambda_seg=self.lambda_seg, lambda_mag=self.lambda_mag, lambda_grad=self.lambda_grad, device=self.device)
        loss = (self.lambda_sim * loss_tensor[0] + self.lambda_seg * loss_tensor[1] + self.lambda_mag * loss_tensor[2] + self.lambda_grad * loss_tensor[3]).float()
        self.log("Global loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Similitude", loss_tensor[0], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Segmentation", self.lambda_seg * loss_tensor[1], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Magnitude", self.lambda_mag * loss_tensor[2], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("Gradient", self.lambda_grad * loss_tensor[3], prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def on_train_epoch_end(self):
        '''
            Compute the dice score on the training dataset
        '''
        self.model.eval()
        if self.current_epoch % 10 == 0:
            for i in self.trainer.train_dataloader.dataset:
                source, target = i.values()
                velocity = self.model(source['image'][tio.DATA].unsqueeze(dim=0).to(self.device),
                                      target['image'][tio.DATA].unsqueeze(dim=0).to(self.device))
                forward_flow, backward_flow = self.model.velocity_to_flow(velocity)

                warped_source_label = self.model.warp(torch.round(source['label'][tio.DATA].float().unsqueeze(dim=0).to(self.device)), forward_flow)
                self.dice_metric(torch.round(warped_source_label), target['label'][tio.DATA].unsqueeze(0).float().to(self.device))
            overall_dice = torch.mean(self.dice_metric.aggregate()).item()
            self.log("Mean dice", overall_dice, prog_bar=True, on_epoch=True, sync_dist=True)
            if self.dice_max < overall_dice:
                self.dice_max = overall_dice
                torch.save(self.model.state_dict(), self.save_path + "/best_model.pth")
        torch.save(self.model.state_dict(), self.save_path + "/last_model.pth")


def train(config):
    config_train = config['train']

    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(config_train['inshape']),
        tio.OneHot(config_train['num_classes'])
    ])

    ## Dateset's configuration : Load a pairwise dataset and the dataloader
    dataset = PairwiseSubjectsDataset(dataset_path=config_train['csv_path'], transform=train_transform, age=False)
    loader = tio.SubjectsLoader(dataset, batch_size=1, num_workers=8, persistent_workers=True)
    in_shape = dataset.dataset[0]['image'][tio.DATA].shape[1:]

    ## Model initialization and weights loading if needed
    try:
        model = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7)
        if "load" in config_train and config_train['load'] != "":
            state_dict = torch.load(config_train['load'])
            model.load_state_dict(state_dict)
    except:
        raise ValueError("Model initialization failed")


    ## Config training with hyperparameters
    trainer_args = {
        'max_epochs': config_train['epochs'],
        'precision': config_train['precision'],
        'strategy': DDPStrategy(find_unused_parameters=True),
        'accumulate_grad_batches': config_train['accumulate_grad_batches'],
        'logger': pl.loggers.TensorBoardLogger(save_dir= "./" + config_train['logger'], name=None)
    }
    trainer_reg = pl.Trainer(**trainer_args)

    save_path = trainer_args['logger'].log_dir.replace(config_train['logger'], "Results")
    create_directory(save_path)



    # Log the config file
    text_md = config_dict_to_markdown(config_train, "Test config")
    trainer_reg.logger.experiment.add_text(text_md, "Test config")
    text_md = config_dict_to_markdown(config['model_reg'], "Registration model config")
    trainer_reg.logger.experiment.add_text(text_md, "Registration model config")
    write_text_to_file(text_md, os.path.join(save_path, "config.md"), mode='w')

    # Train the model
    training_module = RegistrationTrainingModule(model=model,
                                                 lambda_sim=config_train['lam_l'],
                                                 lambda_seg=config_train['lam_s'],
                                                 lambda_mag=config_train['lam_m'],
                                                 lambda_grad=config_train['lam_g'],
                                                 save_path=save_path)
    trainer_reg.fit(training_module, train_dataloaders=loader, val_dataloaders=None)
    torch.save(training_module.model.state_dict(), os.path.join(save_path + "final_model_reg.pth"))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Train Registration 3D Images')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config=config)
