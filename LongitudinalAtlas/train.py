#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import monai.networks.nets
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from dataset.dataset import CustomDataset
from utils.loss import *
import random
import torchvision.transforms.functional as TF
from longitudinal_atlas import LongitudinalAtlas
from utils.utils import normalize_to_0_1

# %% Lightning module


def train_main(args):

    ## Config Dataset / Dataloader
    train_transform = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20)
    ])

    train_dataset = CustomDataset(dataset_path=args.csv_path, t0=args.t0, t1=args.t1, transform=train_transform)
    training_loader = tio.SubjectsLoader(train_dataset, batch_size=1, num_workers=15)

    ## Config model
    # get the spatial dimension of the data (3D)
    in_shape = train_dataset[0].image.shape[1:]
    reg_net = LongitudinalAtlas(
        shape=in_shape,
        loss=args.loss,
        lambda_loss=args.lam_l,
        lambda_seg=args.lam_s,
        lambda_mag=args.lam_m,
        lambda_grad=args.lam_g)
    if args.load:
        reg_net.model.load_state_dict(torch.load(args.load))

    ## Config training
    # %%
    trainer_args = {
        'max_epochs': args.epochs,
        'strategy': DDPStrategy(find_unused_parameters=True),
        'precision': args.precision,
        'accumulate_grad_batches': args.accumulate_grad_batches,
    }

    if args.logger is None:
        trainer_args['logger'] = False
        trainer_args['enable_checkpointing'] = False
    else:
        trainer_args['logger'] = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=args.logger)

    trainer_reg = pl.Trainer(**trainer_args)
    # %%
    trainer_reg.fit(reg_net, training_loader, val_dataloaders=None)
    if args.save:
        torch.save(reg_net.model.state_dict(), args.save)

    template_t0 = None
    for s in train_dataset.subject_dataset:
        if s.age == 0:
            template_t0 = s
            break
    source_data = torch.unsqueeze(template_t0.image.data, 0)
    for i in range(len(train_dataset.subject_dataset)):
        target_subject = train_dataset.subject_dataset[i]
        target_data = torch.unsqueeze(target_subject.image.data, 0)
        weight = target_subject.age - template_t0.age
        svf = reg_net(source_data, target_data)
        flow = reg_net.vecint(weight * svf)
        warped_source = reg_net.transformer(source_data, flow)
        o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=target_subject.image.affine)
        o.save(args.output + '_svf_' + str(i + 1) + '_' + args.loss[0] + '_e' + str(args.epochs) + '.nii.gz')

        img = normalize_to_0_1(warped_source[0].detach())
        trainer_args['logger'].experiment.add_image("Inference Sagittal Plane", TF.rotate(img[:, 64, :, :], 90), i)
        trainer_args['logger'].experiment.add_image("Inference Coronal Plane", TF.rotate(img[:, :, 64, :], 90), i)
        trainer_args['logger'].experiment.add_image("Inference Axial Plane", TF.rotate(img[:, :, :, 64], 90), i)



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False,
                        default="./train_dataset_long.csv")
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required=False, default=22)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required=False, default=35)

    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default=200)
    parser.add_argument('--accumulate_grad_batches', help='Accumulate grad batches', type=int, required=False,
                        default=1)

    parser.add_argument('-l', '--loss', help='Similarity (mse, mae, ncc, lncc)', type=str, action='append',
                        required=False, default=["mse"])
    parser.add_argument('--lam_l', help='Lambda loss for image similarity', type=float, required=False, default=1)
    parser.add_argument('--lam_s', help='Lambda loss for image segmentation', type=float, required=False, default=10)
    parser.add_argument('--lam_m', help='Lambda loss for flow magnitude', type=float, required=False, default=0.01)
    parser.add_argument('--lam_g', help='Lambda loss for flow gradient', type=float, required=False, default=0.1)

    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./save/22_35/")

    parser.add_argument('--load', help='Input model', type=str, required=False)
    parser.add_argument('--save', help='Output model', type=str, required=False, default="./save/22_35/model.pth")

    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required=False,
                        default=32)
    parser.add_argument('--tensor-cores', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    train_main(args=args)


    # o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    # o.save('source.nii.gz')
    # o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    # o.save('target.nii.gz')
