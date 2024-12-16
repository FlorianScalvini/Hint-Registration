import json
import torch
import monai
import argparse
import numpy as np
import torchio as tio
from monai.metrics import DiceMetric
from dataset import PairwiseSubjectsDataset
from Registration import RegistrationModule, RegistrationModuleSVF
from utils import get_cuda_is_available_or_cpu, get_model_from_string

def test(config):
    config_test = config['test']
    device = get_cuda_is_available_or_cpu()

    ## Config Subject
    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(config_test['inshape']),
        tio.OneHot(config_test['num_classes'])
    ])

    dataset = PairwiseSubjectsDataset(dataset_path=config_test['csv_path'], transform=transforms, age=False)

    in_shape = dataset.dataset.shape[1:]

    try:
        model = RegistrationModuleSVF(model=get_model_from_string(config['model_reg']['model'])(**config['model_reg']['args']), inshape=in_shape, int_steps=7).eval().to(device)

        if "load" in config_test and config_test['load'] != "":
            state_dict = torch.load(config_test['load'])
            model.load_state_dict(state_dict)

    except:
        raise ValueError("Model initialization failed")

    dice_metric = DiceMetric(include_background=True, reduction="none").reset()

    for data in dataset:
        source, target = data.values()
        source_img = torch.unsqueeze(source['image'][tio.DATA], 0).to(device)
        target_img = torch.unsqueeze(target['image'][tio.DATA], 0).to(device)
        source_label = torch.unsqueeze(source['label'][tio.DATA], 0).float().to(device)
        target_label = torch.unsqueeze(target['label'][tio.DATA], 0).float().to(device)

        with torch.no_grad():
            forward_flow, backward_flow = model.forward_backward_flow_registration(source_img, target_img)
            wrapped_source_label = model.warp(source_label, forward_flow)
            wrapped_target_label = model.warp(target_label, backward_flow)
            dice_metric(torch.round(wrapped_source_label), target_label)
            dice_metric(torch.round(wrapped_target_label), source_label)

    overall_dice = torch.mean(dice_metric.aggregate())
    print(f"Mean Dice: {torch.mean(overall_dice).item()}")
    print(f"Mean Cortex: {torch.mean(overall_dice[:, 3:5]).item()}")
    print(f"Ventricule: {torch.mean(overall_dice[:,7:9]).item()}")



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Test Registration 3D Images')
    parser.add_argument('--config', type=str, help='Path to the config file', default='./config_test.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    test(config=config)

