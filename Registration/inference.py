import json
import torch
import monai
import argparse
import numpy as np
import torchio as tio
from torch import Tensor
from monai.metrics import DiceMetric
from Registration import RegistrationModuleSVF, RegistrationModule
from utils import get_cuda_is_available_or_cpu


def inference(source : tio.Subject, target : tio.Subject, model : RegistrationModule, device: str):
    model.eval().to(device)
    source_img = source['image'][tio.DATA].unsqueeze(0).to(device)
    target_img = target['image'][tio.DATA].unsqueeze(0).to(device)
    source_label = source['label'][tio.DATA].unsqueeze(0).to(device).float()
    target_label = target['label'][tio.DATA].unsqueeze(0).to(device).float()
    with torch.no_grad():
        forward_flow, backward_flow = model.forward_backward_flow_registration(source_img, target_img)
    warped_source_img = model.warp(source_img, forward_flow)
    warped_target_img = model.warp(target_img, backward_flow)
    warped_source_label = model.warp(source_label, forward_flow)
    warped_target_label = model.warp(target_label, backward_flow)
    return warped_source_img, warped_target_img, warped_source_label, warped_target_label

def main(arguments):
    device = get_cuda_is_available_or_cpu()

    ## Config Subject

    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(arguments.inshape),
        tio.OneHot(arguments.num_classes)
    ])


    source_subject = tio.Subject(
        image=tio.ScalarImage(arguments.image_source),
        label=tio.LabelMap(arguments.label_source),
    )

    target_subject = tio.Subject(
        image=tio.ScalarImage(arguments.image_target),
        label=tio.LabelMap(arguments.target_label),
    )

    reverse_transform = tio.Compose([
        tio.Resize(221),
        tio.CropOrPad(target_shape=source_subject["image"][tio.DATA].shape[1:]),
        tio.OneHot(arguments.num_classes)
    ])

    source_subject = transforms(source_subject)
    target_subject = transforms(target_subject)

    model = RegistrationModuleSVF(
        model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8, 16, 32),
                                                strides=(2, 2)), inshape=source_subject["image"][tio.DATA].shape[1:], int_steps=7).eval().to(device)
    model.load_state_dict(torch.load(arguments.load))

    warped_source_img, warped_target_img, warped_source_label, warped_target_label = inference(source_subject, target_subject, model, device)

    tio.ScalarImage(tensor=warped_source_img.squeeze(0).cpu().detach().numpy(), affine=source_subject['image'].affine).save('./source_warped.nii.gz')
    tio.ScalarImage(tensor=warped_target_img.squeeze(0).cpu().detach().numpy(), affine=target_subject['image'].affine).save('./target_warped.nii.gz')
    tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu().numpy(), affine=source_subject['label'].affine).save('./source_label_warped.nii.gz')
    tio.LabelMap(tensor=torch.argmax(warped_target_label, dim=1).int().detach().cpu().numpy(), affine=target_subject['label'].affine).save('./target_label_warped.nii.gz')

    dice_score = DiceMetric(include_background=True, reduction="none")
    dice = dice_score(torch.round(warped_source_label).cpu(), target_subject["label"][tio.DATA].unsqueeze(0).cpu())[0]

    print(f"Mean Dice: {torch.mean(dice[1:]).item()}")
    print(f"Mean Ventricule Dice: {torch.mean(dice[7:9]).item()}")
    print(f"Mean Cortex Dice: {torch.mean(dice[3:5]).item()}")



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config_inference.json')
    parser.add_argument('--image_source', type=str, help='Path to the source image', required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz")
    parser.add_argument('--image_target', type=str, help='Path to the target image', required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t36.00.nii.gz")
    parser.add_argument('--label_source', type=str, help='Path to the source label', required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz")
    parser.add_argument('--target_label', type=str, help='Path to the target label', required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t36.00_dhcp-19.nii.gz")
    parser.add_argument("--load", type=str, help='Path to the model weights', required=False, default= "./Results/version_39/last_model.pth")
    parser.add_argument("--save_path", type=str, help='Path to save the results', required=False, default="./Results/")
    parser.add_argument("--logger", type=str, help='Logger', required=False, default="log")
    parser.add_argument("--num_classes", type=int, help='Number of classes', required=False, default=20)
    parser.add_argument("--inshape", type=int, help='Input shape', required=False, default=128)

    args = parser.parse_args()
    main(arguments=args)



'''
    subjects_warped_up = reverse_transform(warped_subjects)
    dice = dice_score(torch.argmax(subjects_warped_up["warped_source_label"][tio.DATA].to(device), dim=0),
                      torch.argmax(subject_up["target_label"][tio.DATA].to(device), dim=0), num_classes=20)

    o = tio.ScalarImage(tensor=warped_subjects["warped_source"][tio.DATA].cpu().detach().numpy(),
                        affine=warped_subjects["warped_source"].affine)
    o.save('./source_warped.nii.gz')

    print(f"Mean Dice: {np.mean(dice)}")
    print(f"Mean White Matter Dice: {np.mean(dice[5:7])}")
    print(f"Mean Cortex Dice: {np.mean(dice[3:5])}")

'''
