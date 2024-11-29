import torch
import monai
import argparse
import numpy as np
import torchio as tio

from utils import dice_score, dice_score_old
from Registration import RegistrationModuleSVF, RegistrationModule


def inference(arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Config Subject
    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20)
    ])

    subject = tio.Subject(
        source=tio.ScalarImage(arguments.source),
        target=tio.ScalarImage(arguments.target),
        source_label=tio.LabelMap(arguments.source_label),
        target_label=tio.LabelMap(arguments.target_label)
    )

    transformed_subject = transforms(subject)

    source_img = torch.unsqueeze(transformed_subject['source'][tio.DATA], 0).to(device)
    target_img = torch.unsqueeze(transformed_subject['target'][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(transformed_subject['source_label'][tio.DATA], 0).float().to(device)
    target_label = torch.unsqueeze(transformed_subject['target_label'][tio.DATA], 0).float().to(device)



    model = RegistrationModuleSVF(
        model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8, 16, 32),
                                                strides=(2, 2)), inshape=[128, 128, 128], int_steps=7)

    state_dict = torch.load("/home/florian/Documents/Programs/Hint-Registration/Registration/best_model.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    velocity = model(source_img, target_img)
    forward_flow, backward_flow = model.velocity_to_flow(velocity)

    warped_source_img = model.warp(source_img, forward_flow)
    warped_target_img = model.warp(target_img, backward_flow)
    warped_source_label = model.warp(source_label, forward_flow, mode='nearest')
    warped_target_label = model.warp(target_label, backward_flow, mode='nearest')

    warped_subjects = tio.Subject(
        warped_source=tio.ScalarImage(tensor=warped_source_img[0].detach().cpu(), affine=transformed_subject["source"][tio.AFFINE]),
        warped_target=tio.ScalarImage(tensor=warped_target_img[0].detach().cpu(), affine=transformed_subject["target"][tio.AFFINE]),
        warped_source_label=tio.LabelMap(tensor=warped_source_label[0].detach().cpu(), affine=transformed_subject["source_label"][tio.AFFINE]),
        warped_label_label=tio.LabelMap(tensor=warped_target_label[0].detach().cpu(), affine=transformed_subject["target_label"][tio.AFFINE])
    )
    dice = dice_score(warped_source_label.squeeze(dim=0).detach().cpu().numpy(),
                      target_label.squeeze(dim=0).detach().cpu().numpy())

    print(f"Mean Dice: {np.mean(dice)}")
    print(f"Mean White Matter Dice: {np.mean(dice[5:7])}")
    print(f"Mean Cortex Dice: {np.mean(dice[3:5])}")

    dice = dice_score_old(torch.argmax(warped_source_label.squeeze(dim=0), dim=0),
                          torch.argmax(target_label.squeeze(dim=0), dim=0), num_classes=20)

    print(f"Mean Dice: {np.mean(dice)}")
    print(f"Mean White Matter Dice: {np.mean(dice[5:7])}")
    print(f"Mean Cortex Dice: {np.mean(dice[3:5])}")


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration Inference 3D Images')
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./model_attention_unet_svf.pth")
    parser.add_argument('--output', help='Output directory', type=str, required=False, default="./output/")
    parser.add_argument('-s', '--source', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz")
    parser.add_argument('-t', '--target', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t36.00.nii.gz")
    parser.add_argument('-sl', '--source_label', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz")
    parser.add_argument('-tl', '--target_label', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t36.00_dhcp-19.nii.gz")
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required=False,
                        default=32)
    inference(parser.parse_args())
    print("Success!")