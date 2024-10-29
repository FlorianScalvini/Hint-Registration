import ants
import argparse


def basic_ants_registration(source, target, sourcelabelmap=None, targetlabelmap=None):
    registration_result = ants.registration(target, source, type_of_transform='SyN')
    transform = registration_result['fwdtransforms']
    inverse_transform = registration_result['invtransforms']
    warped_source = ants.apply_transforms(target, source, transform)
    warped_target = ants.apply_transforms(source, target, inverse_transform)

    if sourcelabelmap is None or targetlabelmap is None:
        return warped_source, warped_target
    else:
        warped_source_label = ants.apply_transforms(targetlabelmap, sourcelabelmap, transform, interpolator='nearestNeighbor')
        warped_target_label = ants.apply_transforms(sourcelabelmap, targetlabelmap, inverse_transform, interpolator='nearestNeighbor')

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
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required=False,
                        default="../data/t1-t36.00.nii.gz")
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required=False,
                        default="../data/t1-t21.00.nii.gz")
    parser.add_argument('-tl', '--target_label', help='Target / Reference Image', type=str, required=False,
                        default="../data/t1-t23.00.nii.gz")
    parser.add_argument('-sl', '--source_label', help='Source / Moving Image', type=str, required=False,
                        default="../data/t1-t23.00.nii.gz")
    args = parser.parse_args()
    main(args.source, args.target, args.source_label, args.target_label)
