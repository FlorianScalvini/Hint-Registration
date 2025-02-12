import sys
import json
import torch
import monai
import argparse
import numpy as np

from monai.metrics import DiceMetric
sys.path.insert(0, ".")
sys.path.insert(1, "..")
import torchio2 as tio
from dataset import PairwiseSubjectsDataset
from utils import get_cuda_is_available_or_cpu
from registration_module import RegistrationModule, RegistrationModuleSVF

def test(arguments):
    device = get_cuda_is_available_or_cpu()
    ## Config Subject
    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(arguments.inshape),
        tio.OneHot(arguments.num_classes)
    ])

    ## Dateset's configuration : Load the dataset and the dataloader
    dataset = PairwiseSubjectsDataset(dataset_path=arguments.csv_path, transform=transforms, age=False)
    subject_inshape = dataset[0]['0']['image'][tio.DATA].shape[1:]
    model = RegistrationModuleSVF(
        model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(4, 8, 16, 32),
                                                strides=(2, 2, 2)), inshape=subject_inshape, int_steps=7).eval().to(device)
    model.load_state_dict(torch.load(arguments.load))

    dice_metric = DiceMetric(include_background=True, reduction="none")
    # Test loop
    for data in dataset:
        # Get a pair of images (Source and Target)
        source, target = data.values()
        source_img = torch.unsqueeze(source['image'][tio.DATA], 0).to(device)
        target_img = torch.unsqueeze(target['image'][tio.DATA], 0).to(device)
        source_label = torch.unsqueeze(source['label'][tio.DATA], 0).float().to(device)
        target_label = torch.unsqueeze(target['label'][tio.DATA], 0).float().to(device)

        with torch.no_grad():
            forward_flow, backward_flow = model.forward_backward_flow_registration(source_img, target_img)
            wrapped_source_label = model.warp(source_label, forward_flow)
            wrapped_target_label = model.warp(target_label, backward_flow)
            dice_metric(torch.round(wrapped_source_label), target_label) # Compute the dice score between the Warped Source Label and the Target Label
            dice_metric(torch.round(wrapped_target_label), source_label) # Compute the dice score between the Warped Target Label and the Source Label

    # Compute the global and Cortex mean dice score
    all_dice = dice_metric.get_buffer()
    print(f"Mean Dice: {torch.mean(all_dice[: , 1:]).item()}")
    print(f"Mean Cortex: {torch.mean(all_dice[:, 3:5]).item()}")
    print(f"Ventricule: {torch.mean(all_dice[:,7:9]).item()}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Test Registration 3D Images')
    parser.add_argument("--csv_path", type=str, help="Path to the csv file", required=False, default="./data/full_dataset.csv")
    parser.add_argument("--load", type=str, help="Path to the model weights", required=False, default="./Results/version_32/last_model.pth")
    parser.add_argument("--save_path", type=str, help="Path to save the results", required=False, default="./Results/")
    parser.add_argument("--logger", type=str, help="Logger to use", required=False, default="log")
    parser.add_argument("--inshape", type=int, help="Input shape", required=False, default=128)
    parser.add_argument("--num_classes", type=int, help="Number of classes", required=False, default=20)
    parser.add_argument("--t0", type=int, help="Time at t=0", required=False, default=21)
    parser.add_argument("--t1", type=int, help="Time at t=1", required=False, default=36)

    args = parser.parse_args()
    test(arguments=args)

