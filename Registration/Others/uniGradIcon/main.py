import sys
import torch
import argparse
import torchio2 as tio
from torch import Tensor
from monai.metrics import DiceMetric
from unigradicon import get_unigradicon

sys.path.insert(0, ".")

from icon_registration.mermaidlite import compute_warped_image_multiNC
from Registration.registration_module import RegistrationModule

class UniGradIcon(RegistrationModule):
    def __init__(self):
        super().__init__(model=get_unigradicon(), inshape=[175,175,175])
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        _ = self.model(source, target)
        return self.model.phi_AB_vectorfield

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        _ =  self.forward(source, target)
        return self.model.phi_AB_vectorfield, self.model.phi_BA_vectorfield

    def warp(self, tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return compute_warped_image_multiNC(tensor, flow, self.model.spacing, 0, zero_boundary=True)


def main(source_subject, target_subject):

    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(args.inshape),
        tio.OneHot(args.num_classes)
    ])

    model = UniGradIcon()
    model.cuda()
    source_subject = transforms(source_subject)
    target_subject = transforms(target_subject)
    forward_flow, backward_flow = model.forward_backward_flow_registration(source_subject["image"][tio.DATA].float().unsqueeze(0).cuda(), target_subject["image"][tio.DATA].float().unsqueeze(0).cuda())

    warped_source = model.warp(source_subject["image"][tio.DATA].float().cuda().unsqueeze(0), forward_flow)
    warped_target = model.warp(target_subject["image"][tio.DATA].float().cuda().unsqueeze(0), backward_flow)
    warped_label_source = model.warp(source_subject["label"][tio.DATA].float().cuda().unsqueeze(0), forward_flow)
    warped_label_target = model.warp(target_subject["label"][tio.DATA].float().cuda().unsqueeze(0), backward_flow)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    dice = dice_metric(warped_label_source.detach(), target_subject["label"][tio.DATA].float().cuda().unsqueeze(0))
    print(f"Mean Dice: {torch.mean(dice[0][1:]).cpu().item()}")

    o = tio.ScalarImage(tensor=warped_source.squeeze(dim=0).cpu().detach().numpy(),
                        affine=source_subject["image"].affine)
    o.save('./unigradicon_warped.nii.gz')
    o = tio.ScalarImage(tensor=warped_target.squeeze(dim=0).cpu().detach().numpy(),
                        affine=target_subject["image"].affine)
    o.save('./unigradicon_inverse_warped.nii.gz')
    o = tio.LabelMap(tensor=warped_label_source.squeeze(dim=0).cpu().detach().numpy(),
                     affine=source_subject["label"].affine)
    o.save('./unigradicon_warped_label.nii.gz')
    o = tio.LabelMap(tensor=warped_label_target.squeeze(dim=0).cpu().detach().numpy(),
                     affine=target_subject["label"].affine)
    o.save('./unigradicon_inverse_warped_label.nii.gz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UniGradIcon Registration 3D Image Pair with pretrained network')

    parser.add_argument('-s', '--source', help='Source image', type=str, required=True)
    parser.add_argument('-t', '--target', help='Target image', type=str, required=True)
    parser.add_argument('-sl', '--source_label', help='Source image', type=str, required=True)
    parser.add_argument('-tl', '--target_label', help='Target image', type=str, required=True)
    args = parser.parse_args()
    print(args)

    source_subject = tio.Subject(
        image=tio.ScalarImage(args.source),
        label=tio.LabelMap(args.source_label),
    )
    target_subject = tio.Subject(
        image=tio.ScalarImage(args.target),
        label=tio.LabelMap(args.target_label),
    )

    main(source_subject, target_subject)


