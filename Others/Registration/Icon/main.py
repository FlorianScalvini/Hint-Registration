import itk
import numpy as np
import unittest
import matplotlib.pyplot as plt
import argparse
import icon_registration.itk_wrapper as itk_wrapper
import icon_registration.pretrained_models
import icon_registration.test_utils

def main(source_path, target_path, source_label_path, target_label_path):
    # Define the image type as a 3D image with double precision
    ImageType = itk.D
    source = itk.imread(source_path)
    target = itk.imread(target_path)
    source_label = itk.imread(source_label_path, ImageType)
    target_label = itk.imread(target_label_path, ImageType)

    source = (
        icon_registration.pretrained_models.brain_network_preprocess(
            source
        )
    )
    target = (
        icon_registration.pretrained_models.brain_network_preprocess(
            target
        )
    )

    model = icon_registration.pretrained_models.brain_registration_model(
        pretrained=True
    )

    phi_AB, phi_BA = itk_wrapper.register_pair(model, source, target)

    # itk_wrapper.register_pair uses the spatial metadata of its input images to produce
    # a transform that maps physical coordinates to physical coordinates

    assert isinstance(phi_AB, itk.CompositeTransform)

    interpolator = itk.LinearInterpolateImageFunction.New(source)
    warped_image = itk.resample_image_filter(source,
                                             transform=phi_AB,
                                             interpolator=interpolator,
                                             size=itk.size(target),
                                             output_spacing=itk.spacing(target),
                                             output_direction=target.GetDirection(),
                                             output_origin=target.GetOrigin()
                                             )

    inverse_warped_image = itk.resample_image_filter(target,
                                                     transform=phi_BA,
                                                     interpolator=interpolator,
                                                     size=itk.size(target),
                                                     output_spacing=itk.spacing(target),
                                                     output_direction=target.GetDirection(),
                                                     output_origin=target.GetOrigin()
                                                     )

    inverse_warped_label = itk.resample_image_filter(target_label,
                                                     transform=phi_BA,
                                                     interpolator=interpolator,
                                                     size=itk.size(target),
                                                     output_spacing=itk.spacing(target),
                                                     output_direction=target.GetDirection(),
                                                     output_origin=target.GetOrigin()
                                                     )
    warped_label = itk.resample_image_filter(source_label,
                                             transform=phi_AB,
                                             interpolator=interpolator,
                                             size=itk.size(target),
                                             output_spacing=itk.spacing(target),
                                             output_direction=target.GetDirection(),
                                             output_origin=target.GetOrigin()
                                             )


    itk.imwrite(warped_image, "./icon_warped.nii.gz")
    itk.imwrite(inverse_warped_image, "./icon_inverse_warped.nii.gz")
    itk.imwrite(warped_label, "./icon_warped_label.nii.gz")
    itk.imwrite(inverse_warped_label, "./icon_inverse_warped_label.nii.gz")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='MutliGradIcon Registration 3D Image Pair with pretrained network')
  parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required=False,
                      default="../data/t1-t23.00.nii.gz")
  parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required=False,
                      default="../data/t1-t23.00.nii.gz")
  parser.add_argument('-tl', '--target_label', help='Target / Reference Image', type=str, required=False,
                      default="../data/t1-t23.00.nii.gz")
  parser.add_argument('-sl', '--source_label', help='Source / Moving Image', type=str, required=False,
                      default="../data/t1-t23.00.nii.gz")
  args = parser.parse_args()
  main(args.source, args.target, args.source_label, args.target_label)

