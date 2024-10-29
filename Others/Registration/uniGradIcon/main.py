import torch
import torch.nn.functional as F
from icon_registration.mermaidlite import compute_warped_image_multiNC
import torchio as tio
from unigradicon import get_multigradicon, get_unigradicon
import argparse


def main(subject):
    ### Preprocess
    transform = tio.Compose(
        (
            # Clamp intensities to the 1st and 99th percentiles
            tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
            tio.transforms.CropOrPad(target_shape=192),
            tio.Resize(target_shape=(175, 175, 175))
        )
    )

    subject = transform(subject)

    source = subject["source"][tio.DATA].float().unsqueeze(0)
    target = subject["target"][tio.DATA].float().unsqueeze(0)
    source_label = subject["source_label"][tio.DATA].float().cuda().unsqueeze(0)
    target_label = subject["target_label"][tio.DATA].float().cuda().unsqueeze(0)

    ### Forward Network

    net = get_unigradicon()
    net.cuda()
    net.eval()
    print()
    with torch.no_grad():
        net(source.cuda(), target.cuda())
    ### Wrap label

    warped_label_source = compute_warped_image_multiNC(
        source_label,
        net.phi_AB_vectorfield,
        net.spacing,
        0,
        zero_boundary=True
    )
    warped_label_target = compute_warped_image_multiNC(
        target_label,
        net.phi_BA_vectorfield,
        net.spacing,
        0,
        zero_boundary=True
    )

    o = tio.ScalarImage(tensor=net.warped_image_A.squeeze(dim=0).cpu().detach().numpy(),
                        affine=subject["source"].affine)
    o.save('./unigradicon_warped.nii.gz')
    o = tio.ScalarImage(tensor=net.warped_image_B.squeeze(dim=0).cpu().detach().numpy(),
                        affine=subject["target"].affine)
    o.save('./unigradicon_inverse_warped.nii.gz')
    o = tio.LabelMap(tensor=warped_label_source.squeeze(dim=0).cpu().detach().numpy(),
                     affine=subject["source_label"].affine)
    o.save('./unigradicon_warped_label.nii.gz')
    o = tio.LabelMap(tensor=warped_label_target.squeeze(dim=0).cpu().detach().numpy(),
                     affine=subject["target_label"].affine)
    o.save('./unigradicon_inverse_warped_label.nii.gz')


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
    print(args)

    subject = tio.Subject(
        target=tio.ScalarImage(args.target),
        source=tio.ScalarImage(args.source),
        target_label=tio.LabelMap(args.target_label),
        source_label=tio.LabelMap(args.source_label),
    )

    main(subject=subject)
