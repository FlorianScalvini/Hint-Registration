import ants
import argparse
import shutil
from scipy.io import loadmat

def basic_ants_registration(source, target, sourcelabelmap=None, targetlabelmap=None):
    registration_result = ants.registration(target, source, type_of_transform='SyN')
    transform = registration_result['fwdtransforms']
    inverse_transform = registration_result['invtransforms']
    warped_source = ants.apply_transforms(target, source, transform)
    warped_target = ants.apply_transforms(source, target, inverse_transform)



    warped_source = ants.apply_transforms(fixed=target, moving=source, transformlist=registration_result["fwdtransforms"])
    warped_target = ants.apply_transforms(fixed=source, moving=target, transformlist=registration_result["invtransforms"])
    ants.image_write(warped_source, "./ants_warped_image.nii.gz")
    ants.image_write(warped_target, "./ants_inverse_warped_image.nii.gz")


    # Save forward transforms (warp and affine)
    if isinstance(registration_result["fwdtransforms"], list):  # Multiple transforms
        for i, tf_path in enumerate(registration_result["fwdtransforms"]):
            if tf_path.endswith(".mat"):
                data = loadmat(tf_path)

                # Save content to a .txt file
                with open(f"./fwd_transform_part{i}.txt", "w") as f:
                    for key, value in data.items():
                        if key.startswith("__"):  # Skip metadata keys
                            continue
                        f.write(f"{key}:\n{value}\n")
            else:
                shutil.copy(tf_path, f"./fwd_transform_part{i}.nii.gz")
    else:  # Single transform
        shutil.copy(registration_result["fwdtransforms"], "./fwd_transform.nii.gz" if registration_result["fwdtransforms"].endswith(".nii.gz") else "./fwd_transform.mat")

    # Save forward transforms (warp and affine)
    if isinstance(registration_result["invtransforms"], list):  # Multiple transforms
        for i, tf_path in enumerate(registration_result["invtransforms"]):
            if tf_path.endswith(".mat"):
                # Convert and save .mat files as .txt
                data = loadmat(tf_path)

                # Save content to a .txt file
                with open(f"./inv_transform_part{i}.txt", "w") as f:
                    for key, value in data.items():
                        if key.startswith("__"):  # Skip metadata keys
                            continue
                        f.write(f"{key}:\n{value}\n")
            else:
                shutil.copy(tf_path, f"./inv_transform_part{i}.nii.gz")
    else:  # Single transform
        shutil.copy(registration_result["invtransforms"], "./inv_transform.nii.gz" if registration_result["invtransforms"].endswith(".nii.gz") else "./inv_transform.mat")

    if sourcelabelmap is None or targetlabelmap is None:
        return warped_source, warped_target
    else:
        warped_source_label = ants.apply_transforms(targetlabelmap, sourcelabelmap, registration_result["fwdtransforms"], interpolator='nearestNeighbor')
        warped_target_label = ants.apply_transforms(sourcelabelmap, targetlabelmap, registration_result["invtransforms"], interpolator='nearestNeighbor')

        return warped_source, warped_target, warped_source_label, warped_target_label


def main(source_path, target_path, source_label_path, target_label_path):
    source = ants.image_read(source_path)
    target = ants.image_read(target_path)
    source_label = ants.image_read(source_label_path)
    target_label = ants.image_read(target_label_path)

    warped_source, warped_target, warped_source_label, warped_target_label = \
        basic_ants_registration(source, target, source_label, target_label)

    ants.image_write(warped_source, "ants_warped_image.nii.gz")
    ants.image_write(warped_target, "ants_inverse_warped_image.nii.gz")
    ants.image_write(warped_source_label, "ants_warped_label.nii.gz")
    ants.image_write(warped_target_label, "ants_inverse_warped_label.nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MutliGradIcon Registration 3D Image Pair with pretrained network')
    parser.add_argument('-s', '--source', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz")
    parser.add_argument('-t', '--target', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/structural/t2-t36.00.nii.gz")
    parser.add_argument('-sl', '--source_label', help='Source image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz")
    parser.add_argument('-tl', '--target_label', help='Target image', type=str, required=False, default="/home/florian/Documents/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t36.00_dhcp-19.nii.gz")
    args = parser.parse_args()
    main(args.source, args.target, args.source_label, args.target_label)
