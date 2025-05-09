import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from modules.pairwise_registration import PairwiseRegistrationModuleVelocity
from losses import PairwiseRegistrationLoss, MagnitudeLoss, Grad3d, GradIconInverseConsistency, InverseConsistency, IconInverseConsistency
from modules.data import LongitudinalDataModule, PairwiseRegistrationDataModule
from modules.longitudinal_deformation import OurLongitudinalDeformation
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.nn as nn

import gc
gc.collect()
torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')
    print(OmegaConf.to_yaml(cfg))
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(save_dir, 'log'), name=None, version='')
    loss = PairwiseRegistrationLoss(sim_loss=cfg.train.sim_loss, seg_loss=nn.MSELoss(),mag_loss=MagnitudeLoss(penalty='l2'),
                                    grad_loss=Grad3d(penalty='l2'),inv_loss=GradIconInverseConsistency(),
                                    lambda_sim=cfg.train.lambda_sim, lambda_seg=cfg.train.lambda_seg,
                                    lambda_mag=cfg.train.lambda_mag, lambda_grad=cfg.train.lambda_grad,
                                    lambda_inv=cfg.train.lambda_inv
    )
    model: PairwiseRegistrationModuleVelocity = hydra.utils.instantiate(cfg.pairwise_model)

    if cfg.train.mode == 'longitudinal':
        datamodule: pl.LightningDataModule = LongitudinalDataModule(
            data_dir=cfg.data.csv_path,
            batch_size=cfg.data.batch_size,
            rsize=cfg.data.rsize,
            csize=cfg.data.csize,
            t0=cfg.data.t0,
            t1=cfg.data.t1)
        model: OurLongitudinalDeformation = hydra.utils.instantiate(cfg.longitudinal_model.model, reg_model=model, time_mode=cfg.longitudinal_model.mode, t0=cfg.data.t0, t1=cfg.data.t1, hidden_dim=cfg.longitudinal_model.hidden_dim, max_freq=cfg.longitudinal_model.max_freq, size=cfg.data.rsize)
    else:
        datamodule: pl.LightningDataModule = PairwiseRegistrationDataModule(
            data_dir=cfg.data.csv_path,
            batch_size=cfg.data.batch_size,
            rsize=cfg.data.rsize,
            csize=cfg.data.csize)
    if cfg.train.load != "":
        model.load_state_dict(torch.load(cfg.train.load), strict=False)
    training_module: pl.LightningModule = hydra.utils.instantiate(cfg.train.module, model=model, loss=loss,
                                                                  save_path=save_dir, penalize=cfg.train.penalize,
                                                                  learning_rate=cfg.train.learning_rate)
    trainer = pl.Trainer(max_steps=cfg.train.max_steps, precision=32, num_sanity_val_steps=0, logger=tensorboard_logger,
                         callbacks= [ModelCheckpoint(every_n_train_steps=200, dirpath=save_dir, save_last=True)],
                         check_val_every_n_epoch=10, gradient_clip_val=1., gradient_clip_algorithm='norm')
    checkpoint = None
    if cfg.train.checkpoint != "":
        checkpoint = cfg.train.checkpoint
    trainer.fit(model=training_module,
                datamodule=datamodule,
                ckpt_path=checkpoint)
    print("Training finished")
    print("Saving model")
    training_module.save(save_dir)

if __name__ == '__main__':
    main()