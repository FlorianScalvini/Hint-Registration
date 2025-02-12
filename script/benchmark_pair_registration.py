import csv
import ants
import json
import torch
import monai
import argparse
import numpy as np
import pandas as pd
import torchio2 as tio
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from dataset import WrappedSubjectDataset
from Registration.inference import inference
from Registration.Others.uniGradIcon.main import UniGradIcon
from Registration import RegistrationModule, RegistrationModuleSVF

def _get_transforms(crshape, rshape, num_classes):
    return tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=crshape),
        tio.Resize(rshape),
        tio.OneHot(num_classes)
    ])

def _get_reverse_transform(crshape, rshape, num_classes):
    return tio.Compose([
        tio.Resize(rshape),
        tio.CropOrPad(target_shape=crshape),
        tio.OneHot(num_classes)
    ])

def _compute_registration_dice_between_source_and_targets(model: RegistrationModule, targets : tio.SubjectsDataset, source_subject: tio.Subject,
                                                          transform: tio.Compose, reverse_transform: tio.Compose,device: str):

    source_subject_transformed = transform(source_subject)
    dice_scores = []
    dice_metric = DiceMetric(include_background=True, reduction="none")
    for subject in targets:
        transformed_subject = transform(subject)
        _, _, warped_source_label, _ = inference(source_subject_transformed, transformed_subject, model, device)
        warped_source_label = tio.LabelMap(tensor=torch.argmax(torch.round(warped_source_label), dim=1).int().detach().cpu(), affine=transformed_subject['label'].affine)
        warped_source_label = reverse_transform(warped_source_label)[tio.DATA].float().cpu().unsqueeze(0).to(device)
        target_label = subject["label"][tio.DATA].float().to(device).unsqueeze(0)
        dice = dice_metric(warped_source_label, target_label)[0]
        dice_metric.reset()
        dice_scores.append(dice.cpu().numpy())
    return np.vstack(dice_scores)

def _compute_voxelmorph_like(dataset_path: str, source_image_path: str, source_label_path: str, model_path: str,
                             crshape: list[int], rshape:list[int], num_classes: int, device: str):
    source_subject = tio.Subject(
        image=tio.ScalarImage(source_image_path),
        label=tio.LabelMap(source_label_path)
    )
    transform = _get_transforms(crshape=crshape, rshape=rshape, num_classes=num_classes)
    reverse_transform = _get_reverse_transform(crshape=source_subject["image"][tio.DATA].shape[1:], rshape=crshape, num_classes=num_classes)
    targets = WrappedSubjectDataset(dataset_path=dataset_path, transform=None)
    inshape = transform(source_subject["image"][tio.DATA]).shape[1:]
    model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(4, 8, 16, 32), strides=(2,2, 2)), inshape=inshape, int_steps=7).eval().to(device)
    model.load_state_dict(torch.load(model_path))
    return _compute_registration_dice_between_source_and_targets(model=model, targets=targets, source_subject=source_subject,
                                                                 transform=transform, reverse_transform=reverse_transform,
                                                                 device=device)

def _compute_unigradicon(dataset_path, source_image_path, source_label_path, device, num_classes):
    source_subject = tio.Subject(
        image=tio.ScalarImage(source_image_path),
        label=tio.LabelMap(source_label_path)
    )
    transform = _get_transforms(crshape=(180, 221, 180), rshape=(175, 175, 175), num_classes=num_classes)
    reverse_transform = _get_reverse_transform(crshape=source_subject["image"][tio.DATA].shape[1:], rshape=(180, 221, 180), num_classes=num_classes)
    targets = WrappedSubjectDataset(dataset_path=dataset_path, transform=None)
    model = UniGradIcon()
    return _compute_registration_dice_between_source_and_targets(model=model, targets=targets, source_subject=source_subject,
                                                                 transform=transform, reverse_transform=reverse_transform,
                                                                 device=device)

def _compute_ants(dataset_path, source_image_path, source_label_path, num_classes):
    targets = []
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        targets.append((row['image'], row['label']))
    source_image = ants.image_read(source_image_path)
    source_label = ants.image_read(source_label_path)
    dice_metric = DiceMetric(include_background=True, reduction="none")
    dice_scores = []
    for target_image, target_label in targets:
        target_image = ants.image_read(target_image)
        target_label = ants.image_read(target_label)
        registration_result = ants.registration(target_image, source_image,
                                                type_of_transform='SyN',
                                                reg_iterations=(100, 40, 20),
                                                verbose=False,
                                                aff_iterations=(0,0,0,0))
        warped_source_label = ants.apply_transforms(target_label, source_label,
                                                    registration_result["fwdtransforms"],
                                                    interpolator='nearestNeighbor')
        one_hot_warped_source_label = np.eye(num_classes)[warped_source_label.numpy().astype(int)].transpose(3, 0, 1, 2)
        one_hot_target_label = np.eye(num_classes)[target_label.numpy().astype(int)].transpose(3, 0, 1, 2)
        dice = dice_metric(torch.tensor(one_hot_warped_source_label).unsqueeze(0),
                           torch.tensor(one_hot_target_label).unsqueeze(0))[0]
        dice_scores.append(dice.cpu().numpy())
    return np.vstack(dice_scores)

def main(arguments):
    scores = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if arguments.unigrad:
        scores["Unigradicon"] = _compute_unigradicon(dataset_path=arguments.target,
                                                     source_image_path=arguments.source,
                                                     source_label_path=arguments.source_label,
                                                     device=device, num_classes=arguments.num_classes)
    if arguments.ants:
        scores["ANTS"] = _compute_ants(dataset_path=arguments.target,
                                       source_label_path=arguments.source,
                                       source_image_path=arguments.source_label,
                                       num_classes=arguments.num_classes)
    if arguments.voxelmorph:
        scores["Voxelmorph-like"] = _compute_voxelmorph_like(dataset_path=arguments.target,
                                                             source_image_path=arguments.source,
                                                             source_label_path=arguments.source_label,
                                                             model_path=arguments.load,
                                                             rshape=arguments.vsize,
                                                             crshape=arguments.vcrop,
                                                             num_classes=arguments.num_classes,
                                                             device=device)
    with open("./results_pairwise.csv", mode='w') as file:
        header = ["index", "Unigradicon", "ANTS", "Voxelmorph-like"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for i in range(len(scores[next(iter(scores))])):
            save = {"index": i}
            for key in scores.keys():
                save[key] = scores[key][i]
            writer.writerow(save)
    color = ["magenta", "peru", "green"]
    marker = ["D", "h", "x"]
    x = [i for i in range(len(scores[next(iter(scores))]))]
    for i in range(len(scores)):
        label = list(scores.keys())[i]
        plt.plot(x[1:],  np.mean(scores[label][1:, 1:], axis=1).tolist(), color=color[i], marker=marker[i], label=label)
        plt.legend()
    plt.show()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')
    parser.add_argument('--unigrad', type=bool, help='Unigradicon benchmark', required=False, default=False)
    parser.add_argument('--ants', type=bool, help='ANTS benchmark', required=False, default=False)
    parser.add_argument('--voxelmorph', type=bool, help='Voxelmorph-like benchmark', required=False, default=True)
    parser.add_argument('--source', type=str, help='Path to the source image', required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz")
    parser.add_argument('--source_label', type=str, help='Path to the source image', required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz")
    parser.add_argument('--target', type=str, help='Path to the target images/labels', required=False, default="/home/florian/Documents/Programs/Hint-Registration/data/dHCP/dataset.csv")
    parser.add_argument('--num_classes', type=int, help='Number of classes', required=False, default=20)
    # Paramaters for Voxelmorph-like method
    parser.add_argument('--load', type=str, help='VoxelMorph-like network model path', required=False, default="/home/florian/Documents/Programs/Hint-Registration/Registration/Results/version_76/last_model.pth")
    parser.add_argument('--vcrop', type=int, help='Crop shape', nargs='+', required=False, default=221)
    parser.add_argument('--vsize', type=int, help='VoxelMorph-like network voxel size', nargs='+', required=False, default=128)
    args = parser.parse_args()
    main(args)
