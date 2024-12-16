import json
import torch
import monai
import argparse
import numpy as np
import torchio as tio
from monai.metrics import DiceMetric
from Registration import RegistrationModuleSVF, RegistrationModule
from utils import get_cuda_is_available_or_cpu, config_dict_to_tensorboard, get_model_from_string

def inference(config):
    config_inference = config['inference']
    device = get_cuda_is_available_or_cpu()

    ## Config Subject
    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(config_inference['inshape']),
        tio.OneHot(config_inference['num_classes'])
    ])

    subject = tio.Subject(
        source=tio.ScalarImage(config_inference["image_source"]),
        target=tio.ScalarImage(config_inference["image_target"]),
        source_label=tio.LabelMap(config_inference["label_source"]),
        target_label=tio.LabelMap(config_inference["label_target"])
    )

    transformed_subject = transforms(subject)

    source_img = torch.unsqueeze(transformed_subject['source'][tio.DATA], 0).to(device)
    target_img = torch.unsqueeze(transformed_subject['target'][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(transformed_subject['source_label'][tio.DATA], 0).float().to(device)
    target_label = torch.unsqueeze(transformed_subject['target_label'][tio.DATA], 0).float().to(device)
    in_shape = source_img.shape[2:]

    try:
        model = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7).eval().to(device)
        if "load" in config_inference and config_inference['load'] != "":
            state_dict = torch.load(config_inference['load'])
            model.load_state_dict(state_dict)

    except:
        raise ValueError("Model initialization failed")


    velocity = model(source_img, target_img)
    forward_flow, backward_flow = model.velocity_to_flow(velocity)
    warped_source_img = model.warp(source_img, forward_flow)
    warped_target_img = model.warp(target_img, backward_flow)
    warped_source_label = model.warp(source_label, forward_flow)
    warped_target_label = model.warp(target_label, backward_flow)

    tio.ScalarImage(tensor=warped_source_img.squeeze(0).cpu().detach().numpy(), affine=transformed_subject['source'].affine).save('./source_warped.nii.gz')
    tio.ScalarImage(tensor=warped_target_img.squeeze(0).cpu().detach().numpy(), affine=transformed_subject['target'].affine).save('./target_warped.nii.gz')
    tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).int().detach().cpu().numpy(), affine=transformed_subject['source_label'].affine).save('./source_label_warped.nii.gz')
    tio.LabelMap(tensor=torch.argmax(warped_target_label, dim=1).int().detach().cpu().numpy(), affine=transformed_subject['target_label'].affine).save('./target_label_warped.nii.gz')

    dice_score = DiceMetric(include_background=True, reduction="none")
    dice = dice_score(torch.round(warped_source_label), target_label)[0]
    print(f"Mean Dice: {torch.mean(dice).item()}")
    print(f"Mean Ventricule Dice: {torch.mean(dice[7:9]).item()}")
    print(f"Mean Cortex Dice: {torch.mean(dice[3:5]).item()}")



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config_inference.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    inference(config=config)



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
