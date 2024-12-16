import os
import csv
import json
import monai
import torch
import argparse
import itertools
import numpy as np
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from dataset import subjects_from_csv
from Registration import RegistrationModuleSVF
import torchvision.transforms.functional as TF
from utils import get_cuda_is_available_or_cpu, normalize_to_0_1, create_directory, config_dict_to_markdown, write_text_to_file, map_labels_to_colors, seg_map_error


NUM_CLASSES = 20

def test(config):
    # Config Dataset / Dataloader
    config_test = config['test']
    device = get_cuda_is_available_or_cpu()
    loggers = pl.loggers.TensorBoardLogger(save_dir= "./" + config_test['logger'], name=None)
    save_path = loggers.log_dir.replace(config_test['logger'], "Results")
    create_directory(save_path)

    text_md = config_dict_to_markdown(config['test'], "Test config")
    loggers.experiment.add_text(text_md)
    text_md = config_dict_to_markdown(config['model_reg'], "Registration model config")
    loggers.experiment.add_text(text_md)
    write_text_to_file(text_md, os.path.join(save_path, "config.md"), mode='w')

    transforms = [
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(NUM_CLASSES)
    ]

    subjects_original_shape = (180, 221, 180)

    reverse_transforms =tio.Compose([
        tio.Resize(221),
        tio.CropOrPad(target_shape=subjects_original_shape)
    ])

    transforms_onehot = tio.Compose([
        tio.OneHot(20)]
    )

    subjects = subjects_from_csv(config_test['csv_path'], age=True, lambda_age=lambda x: (x - config_test['t0']) / (config_test['t1'] - config_test['t0']))
    subjects_set = tio.SubjectsDataset(subjects, transform=tio.Compose(transforms=transforms))
    transformationPairs = list(itertools.combinations(range(len(subjects_set)), 2))
    in_shape = subjects_set[0]['image'][tio.DATA].shape[1:]
    subjects_set_orig = tio.SubjectsDataset(subjects, transform=None)
    model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=in_shape, int_steps=7).to(device)
    try:
        state_dict = torch.load(config_test['load'])
        model.load_state_dict(state_dict)
    except:
        raise ValueError("No model to load or model not compatible")

    denum = 0
    num = torch.zeros([1, 3] + list(in_shape)).to(device)

    with torch.no_grad():
        for i, j in transformationPairs:
            sample_i = subjects_set[i]
            sample_j = subjects_set[j]
            time_ij = sample_j['age'] - sample_i['age']
            velocity_ij = model(sample_i['image'][tio.DATA].float().unsqueeze(dim=0).to(device),
                                sample_j['image'][tio.DATA].unsqueeze(dim=0).float().to(device))
            num += velocity_ij * time_ij
            denum += time_ij * time_ij
        weightedMeanVelocity = num / denum if denum != 0 else torch.zeros_like(num)


    source = None

    for s in subjects_set:
        if s.age == 0:
            source = s
            break

    source_image = torch.unsqueeze(source["image"][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(source["label"][tio.DATA], 0).to(device)


    dice_score = DiceMetric(include_background=False, reduction="none")

    with open(save_path + "/results.csv", mode='w') as file:
        header = ["time", "mDice", "Cortex", "Ventricule", "all"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for i in range(len(subjects_set)):
            other_target = subjects_set[i]
            age = int(other_target['age'] * (config_test['t1'] - config_test['t0']) + config_test['t0'])
            diff_age = other_target['age'] - source['age']
            weighted_velocity = weightedMeanVelocity * diff_age
            forward_flow, backward_flow = model.velocity_to_flow(velocity=weighted_velocity)
            warped_source_label = model.warp(source_label.float(), forward_flow)
            warped_source_image = model.warp(source_image.float(), forward_flow)

            new_subject = tio.Subject(
                image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0),
                                      affine=source['image'].affine),
                label=tio.LabelMap(tensor=torch.argmax(torch.round(warped_source_label), dim=1).int().detach().cpu(),
                                   affine=source['label'].affine)
            )
            new_subject = reverse_transforms(new_subject)
            label_map = new_subject['label']
            label_map.save(save_path + "/" + str(i) + "_fused_source_label.nii.gz")
            image_wrap = new_subject['image']
            image_wrap.save(save_path + "/" + str(i) + "_fused_source_image.nii.gz")

            new_subject = transforms_onehot(new_subject)
            target_inter = transforms_onehot(subjects_set_orig[i])

            colored_error_seg = map_labels_to_colors(
                seg_map_error(new_subject["label"][tio.DATA].unsqueeze(0), target_inter["label"][tio.DATA].unsqueeze(0),
                              dim=1)).squeeze().permute(3, 0, 1, 2)

            loggers.experiment.add_image("Atlas Sagittal Plane",
                                         TF.rotate(colored_error_seg[:, int(in_shape[0] / 2), :, :], 90), age)
            loggers.experiment.add_image("Atlas Coronal Plane",
                                         TF.rotate(colored_error_seg[:, :, int(in_shape[1] / 2), :], 90), age)
            loggers.experiment.add_image("Atlas Axial Plane",
                                         TF.rotate(colored_error_seg[:, :, :, int(in_shape[2] / 2)], 90), age)

            dice = dice_score(new_subject["label"][tio.DATA].unsqueeze(0).to(device),
                              target_inter["label"][tio.DATA].unsqueeze(0).float().to(device))

            """
            warped_source_image = reg_net.warp(source_image.float(), forward_flow)
            o = tio.ScalarImage(tensor=warped_source_image.detach().numpy(), affine=subjects_set[0]['source'].affine)
            o.save(os.path.join(save_path, str(real_age) + "_image.noo.gz"))
            """
            writer.writerow({
                "time": age,
                "mDice": torch.mean(dice[0]).item(),
                "Cortex": torch.mean(dice[0][3:5]).item(),
                "Ventricule": torch.mean(dice[0][7:9]).item(),
                "all": dice[0].cpu().numpy()
            })
            loggers.experiment.add_scalar("Dice ventricule", torch.mean(dice[0][7:9]).item(), age)
            loggers.experiment.add_scalar("Dice cortex", torch.mean(dice[0][3:5]).item(), age)
            loggers.experiment.add_scalar("mDice", torch.mean(dice[0]).item(), age)
            print(age, torch.mean(dice).item())


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with MLP model')
    parser.add_argument('--config', type=str, help='Path to the config file', default='/home/florian/Documents/Programs/Hint-Registration/TemporalTrajectory/SVF/Linear/config_test.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    test(config=config)
