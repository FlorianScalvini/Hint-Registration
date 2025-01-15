import ants
import argparse
import numpy as np
from monai.metrics import DiceMetric



def ants_registration(source, target, source_label, target_label):
    source = ants.image_read(source)
    target = ants.image_read(target)
    source_label = ants.image_read(source_label)
    target_label = ants.image_read(target_label)

    registration_result = ants.registration(target, source, type_of_transform='SyN')
    warped_source = ants.apply_transforms(fixed=target, moving=source,
                                          transformlist=registration_result["fwdtransforms"])
    warped_target = ants.apply_transforms(fixed=source, moving=target,
                                          transformlist=registration_result["invtransforms"])

    warped_source_label = ants.apply_transforms(target_label, source_label, registration_result["fwdtransforms"], interpolator='nearestNeighbor')
    warped_target_label = ants.apply_transforms(source_label, target_label, registration_result["invtransforms"],
                                                interpolator='nearestNeighbor')
    return warped_source, warped_target, warped_source_label, warped_target_label

def main(source, target, source_label, target_label, num_classes=20):
    warped_source, warped_target, warped_source_label, warped_target_label = ants_registration(source, target, source_label, target_label)

    ants.image_write(warped_source, "./ants_warped_image.nii.gz")
    ants.image_write(warped_target, "./ants_inverse_warped_image.nii.gz")
    ants.image_write(warped_source, "ants_warped_image.nii.gz")
    ants.image_write(warped_target, "ants_inverse_warped_image.nii.gz")
    ants.image_write(warped_source_label, "ants_warped_label.nii.gz")
    ants.image_write(warped_target_label, "ants_inverse_warped_label.nii.gz")

    dice_metric = DiceMetric(include_background=True, reduction="none")
    one_hot_warped_source_label = np.eye(num_classes)[warped_source_label.numpy().astype(int)].transpose(3, 0, 1, 2)
    one_hot_target_label = np.eye(num_classes)[target_label.numpy().astype(int)].transpose(3, 0, 1, 2)
    dice = dice_metric(one_hot_warped_source_label.numpy(), one_hot_target_label.numpy())[0][1:]
    print(dice)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MutliGradIcon Registration 3D Image Pair with pretrained network')
    parser.add_argument('-s', '--source', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz")
    parser.add_argument('-t', '--target', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t36.00.nii.gz")
    parser.add_argument('-sl', '--source_label', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz")
    parser.add_argument('-tl', '--target_label', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t36.00_dhcp-19.nii.gz")
    args = parser.parse_args()
    main(args.source, args.target, args.source_label, args.target_label)
