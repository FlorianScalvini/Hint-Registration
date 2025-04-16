import os.path

import torch
import monai
import argparse
import torchio as tio
from monai.metrics import DiceMetric
from modules.pairwise_registration import PairwiseRegistrationModuleVelocity
from losses.jacobian import calculate_jacobian
from utils import get_cuda_is_available_or_cpu


def inference(source : tio.Subject, target : tio.Subject, model : PairwiseRegistrationModuleVelocity, device: str):
    model.eval().to(device)
    source_img = source['image'][tio.DATA].unsqueeze(0).to(device)
    target_img = target['image'][tio.DATA].unsqueeze(0).to(device)
    source_label = source['label'][tio.DATA].unsqueeze(0).to(device).float()
    target_label = target['label'][tio.DATA].unsqueeze(0).to(device).float()
    with torch.no_grad():
        velocity = model(source_img, target_img)
    forward_flow = model.velocity2displacement(velocity)
    backward_flow = model.velocity2displacement(-velocity)
    warped_source_img = model.warp(source_img, forward_flow)
    warped_target_img = model.warp(target_img, backward_flow)
    warped_source_label = model.warp(source_label, forward_flow)
    warped_target_label = model.warp(target_label, backward_flow)
    return warped_source_img, warped_target_img, warped_source_label, warped_target_label, forward_flow, backward_flow

def main(arguments):
    device = get_cuda_is_available_or_cpu()

    ## Config Subject
    source_subject = tio.Subject(
        image=tio.ScalarImage(arguments.image_source),
        label=tio.LabelMap(arguments.label_source),
    )

    target_subject = tio.Subject(
        image=tio.ScalarImage(arguments.image_target),
        label=tio.LabelMap(arguments.target_label),
    )

    transforms = tio.Compose([
        tio.CropOrPad(target_shape=args.csize),
        tio.Resize(target_shape=args.rsize),
        tio.ZNormalization(masking_method='label'),
        tio.OneHot(args.num_classes)
    ])

    reverse_transform = tio.Compose([
        tio.Resize(target_shape=args.csize),
        tio.CropOrPad(target_shape=source_subject["image"][tio.DATA].shape[1:]),
        tio.OneHot(args.num_classes)
    ])

    source_subject_transformed = transforms(source_subject)
    target_subject_transformed = transforms(target_subject)

    model = PairwiseRegistrationModuleVelocity(
        model=monai.networks.nets.AttentionUnet(
            dropout=0.1, spatial_dims=3, in_channels=2, out_channels=3, channels=[8, 16, 32, 64], strides=[2,2,2]),
        int_steps=7).eval().to(device)
    model.load_state_dict(torch.load(arguments.load))

    warped_source_img, warped_target_img, warped_source_label, warped_target_label, forward_flow, backward_flow = inference(source_subject_transformed, target_subject_transformed, model, device)
    warped_subjects = tio.Subject(
        warped_source_img = tio.ScalarImage(tensor=warped_source_img.squeeze(0).cpu().detach(), affine=source_subject_transformed['image'].affine),
        warped_target_img = tio.ScalarImage(tensor=warped_target_img.squeeze(0).cpu().detach(), affine=target_subject_transformed['image'].affine),
        warped_source_label = tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu(), affine=source_subject_transformed['label'].affine),
        warped_target_label = tio.LabelMap(tensor=torch.argmax(warped_target_label, dim=1).int().detach().cpu(), affine=target_subject_transformed['label'].affine)
    )


    # Compute the determinant for each voxel
    jacobian_det = torch.linalg.det(calculate_jacobian(forward_flow, spacing=(1, 1, 1)))
    negative_mask = jacobian_det < 0
    print(f"Nb negative Det(J): {negative_mask.sum().item()}")

    tio.ScalarImage(tensor=forward_flow.squeeze(0).cpu().detach(), affine=source_subject_transformed['image'].affine).save(os.path.join(args.output_dir, './forward_flow.nii.gz'))
    tio.ScalarImage(tensor=backward_flow.squeeze(0).cpu().detach(), affine=target_subject_transformed['image'].affine).save(os.path.join(args.output_dir, './backward_flow.nii.gz'))
    tio.ScalarImage(tensor=jacobian_det.cpu().detach(), affine=source_subject_transformed['image'].affine).save(os.path.join(args.output_dir, './det_J.nii.gz'))
    warped_subjects["warped_source_img"].save(os.path.join(args.output_dir, './source_warped.nii.gz'))
    warped_subjects["warped_target_img"].save(os.path.join(args.output_dir, './target_warped.nii.gz'))
    warped_subjects["warped_source_label"].save(os.path.join(args.output_dir, './source_label_warped.nii.gz'))
    warped_subjects["warped_target_label"].save(os.path.join(args.output_dir, './target_label_warped.nii.gz'))

    warped_subjects = reverse_transform(warped_subjects)

    dice_score = DiceMetric(include_background=True, reduction="none", num_classes=args.num_classes)
    dice = dice_score(warped_subjects["warped_source_label"][tio.DATA].unsqueeze(0).cpu(), target_subject["label"][tio.DATA].int().unsqueeze(0).cpu())[0]
    print(f"Mean Dice: {torch.mean(dice[1:]).item()}")
    print(f"Mean Ventricule Dice: {torch.mean(dice[7:9]).item()}")
    print(f"Mean Cortex Dice: {torch.mean(dice[3:5]).item()}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')
    parser.add_argument('--image_source', type=str, help='Path to the source image', required=True)
    parser.add_argument('--image_target', type=str, help='Path to the target image', required=True)
    parser.add_argument('--label_source', type=str, help='Path to the source label', required=True)
    parser.add_argument('--target_label', type=str, help='Path to the target label', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to the output directory', required=True)
    parser.add_argument("--load", type=str, help='Path to the model weights', required=True)
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize shape', default=[128, 128, 128])
    parser.add_argument('--csize', type=int, nargs='+', help='Cropsize shape', default=[221, 221, 221])
    parser.add_argument("--num_classes", type=int, help='Number of classes', required=False, default=20)
    args = parser.parse_args()
    main(arguments=args)
