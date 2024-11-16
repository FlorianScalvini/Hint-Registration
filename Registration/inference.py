import torch
import monai
import argparse
import numpy as np
import torchio as tio

from utils import dice_score
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

    reverse_transform = tio.Compose([
        tio.Resize(221)
    ])

    subject = tio.Subject(
        source=tio.ScalarImage(arguments.source),
        target=tio.ScalarImage(arguments.target),
        source_label=tio.LabelMap(arguments.source_label),
        target_label=tio.LabelMap(arguments.target_label)
    )

    transformed_subject = transforms(subject)


    model = RegistrationModuleSVF(
        model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8, 16, 32),
                                                strides=(2, 2)), inshape=[128, 128, 128], int_steps=7)
    try:
        model.model.load_state_dict(torch.load(arguments.load))
    except:
        raise FileNotFoundError("No model to load")

    model.eval().to(device)

    source_img = torch.unsqueeze(transformed_subject['source'][tio.DATA], 0).to(device)
    target_img = torch.unsqueeze(transformed_subject['target'][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(transformed_subject['source_label'][tio.DATA], 0).float().to(device)
    target_label = torch.unsqueeze(transformed_subject['target_label'][tio.DATA], 0).float().to(device)

    with torch.no_grad():
        forward_flow, backward_flow = model.forward_backward_flow_registration(source_img, target_img)

    wrapped_source_img = model.wrap(source_img, forward_flow)
    wrapped_target_img = model.wrap(target_img, backward_flow)

    wrapped_source_label = model.wrap(source_label, forward_flow)
    dice = dice_score(torch.argmax(wrapped_source_label, dim=1),
                      torch.argmax(target_label, dim=0), num_classes=20)

    wrapped_reversed_transformed_source_img = reverse_transform(wrapped_source_img)
    wrapped_reversed_transformed_source_target = reverse_transform(wrapped_target_img)

    o = tio.ScalarImage(tensor=wrapped_reversed_transformed_source_img[0].detach().numpy(), affine=subject.source.affine)
    o.save(arguments.output+'_warped.nii.gz')
    o = tio.ScalarImage(tensor=wrapped_reversed_transformed_source_target[0].detach().numpy(), affine=subject.target.affine)
    o.save(arguments.output+'_inverse_warped.nii.gz')

    print(f"Mean Dice: {np.mean(dice)}")
    print(f"Mean White Matter Dice: {np.mean(dice[5:7])}")
    print(f"Mean Cortex Dice: {np.mean(dice[3:5])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration Inference 3D Images')
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./model.pth")
    parser.add_argument('--output', help='Output directory', type=str, required=False, default="./output/")
    parser.add_argument('-s', '--source', help='Source image', type=str, required=False, default="./home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t24.00.nii.gz")
    parser.add_argument('-t', '--target', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t28.00.nii.gz")
    parser.add_argument('-sl', '--source_label', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t24.00_dhcp-19.nii.gz")
    parser.add_argument('-tl', '--target_label', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t24.00_dhcp-19.nii.gz")
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required=False,
                        default=32)
    inference(parser.parse_args())
    print("Success!")