#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torchio as tio
import pytorch_lightning as pl

from utils.loss import *
import torchvision.transforms.functional as TF
from utils.metric import dice_score
from longitudinal_atlas import LongitudinalAtlas
from dataset.dataset import CustomDataset

def normalize_to_0_1(volume):
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)


def inference(arguments):

    df = pd.read_csv(arguments.tsv_file, sep=',')
    subjects = []
    for index, row in df.iterrows():
        subject = tio.Subject(
            image=tio.ScalarImage(row['image']),
            label=tio.LabelMap(row['label']),
            age= (row["age"] - arguments.t0) / (arguments.t1 - arguments.t0)
        )
        subjects.append(subject)

    train_transform = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
    ]
    training_set = tio.SubjectsDataset(subjects, transform=tio.Compose(train_transform))
    in_shape = training_set[0].image.shape[1:]

    reg_net = LongitudinalAtlas(
        shape=in_shape,
        loss=args.loss,
        lambda_loss=args.lam_l,
        lambda_mag=args.lam_m,
        lambda_grad=args.lam_g)

    reg_net.model.load_state_dict(torch.load(arguments.load))
    loggers = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=arguments.logger)

    template_t0 = None
    for s in training_set:
        if s.age == 0:
            template_t0 = s
            break

    source_data = torch.unsqueeze(template_t0.image.data, 0)
    source_data_label = torch.unsqueeze(template_t0.label.data, 0)
    for i in range(len(subjects)):
        target_subject = training_set[i]
        target_image = torch.unsqueeze(target_subject.image.data, 0)
        target_label = torch.unsqueeze(target_subject.label.data.contiguous(), 0)
        weight = target_subject.age - template_t0.age
        svf = reg_net(source_data, target_image)
        flow = reg_net.vecint(weight * svf)
        warped_source = reg_net.transformer(source_data, flow)
        warped_source_label = reg_net.transformer(source_data_label.float(), flow)

        o = tio.LabelMap(tensor=warped_source_label[0].detach().numpy(), affine=target_subject.image.affine)
        o.save(arguments.output + '_svf_' + str(i + 1) + '_label' + '.nii.gz')
        o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=target_subject.image.affine)
        o.save(arguments.output + '_svf_' + str(i + 1) +  '_image' + '.nii.gz')

        img = normalize_to_0_1(warped_source[0].detach())
        loggers.experiment.add_image("Test Sagittal Plane", TF.rotate(img[:, 64, :, :], 90), i)
        loggers.experiment.add_image("Test Coronal Plane", TF.rotate(img[:, :, 64, :], 90), i)
        loggers.experiment.add_image("Test Axial Plane", TF.rotate(img[:, :, :, 64], 90), i)


        dice = dice_score(warped_source_label, target_label, num_classes=20)
        loggers.experiment.add_scalar("Dice white matter", np.mean(dice[5:7]), i)
        loggers.experiment.add_scalar("Dice cortex", np.mean(dice[3:5]), i)
        loggers.experiment.add_scalar("mDice", np.mean(dice), i)



# %% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Registration 3D Longitudinal Images : Inference')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False,
                        default="./train_dataset_long.csv")
    parser.add_argument('--t0', help='Initial time (t0) in week', type=float, required=False, default=22)
    parser.add_argument('--t1', help='Final time (t1) in week', type=float, required=False, default=35)

    parser.add_argument('-o', '--output', help='Output filename', type=str, required=False, default="./save/test/")
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./save/22_35/model.pth")
    parser.add_argument('--logger', help='Logger name', type=str, required=False, default="log")


    args = parser.parse_args()

    if args.tensor_cores:
        torch.set_float32_matmul_precision('high')

    inference(arguments=args)
