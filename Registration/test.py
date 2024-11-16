import torch
import monai
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torchio as tio
import pytorch_lightning as pl
import torchvision.transforms.functional as TF

from utils import dice_score
from dataset import PairwiseSubjectsDataset
from Registration import RegistrationModule, RegistrationModuleSVF

def main(arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Config Dataset / Dataloader
    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
        tio.transforms.Clamp(out_min=0, out_max=1),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(128),
        tio.OneHot(20)
    ])

    dataset = PairwiseSubjectsDataset(dataset_path=arguments.csv_path, transform=transforms)

    model = RegistrationModuleSVF(model=monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2)), inshape=[128, 128, 128], int_steps=7)
    try:
        model.model.load_state_dict(torch.load(arguments.load))
    except:
        raise FileNotFoundError("No model to load")

    model.eval().to(device)
    mean_dice = np.array([])
    mean_white_matter_dice = np.array([])
    mean_cortex_dice = np.array([])

    for source, target in dataset:
        source_img = torch.unsqueeze(source['image'][tio.DATA], 0).to(device)
        target_img = torch.unsqueeze(target['image'][tio.DATA], 0).to(device)
        source_label = torch.unsqueeze(source['label'][tio.DATA], 0).float().to(device)
        target_label = torch.unsqueeze(target['label'][tio.DATA], 0).float().to(device)

        with torch.no_grad():
            forward_flow, backward_flow = model.forward_backward_flow_registration(source_img, target_img)
        wrapped_source_label = model.wrap(source_label, forward_flow)
        dice = dice_score(torch.argmax(wrapped_source_label, dim=1),
                          torch.argmax(target_label, dim=0), num_classes=20)
        mean_dice = np.append(mean_dice, np.mean(dice))
        mean_white_matter_dice = np.append(mean_white_matter_dice, np.mean(dice[5:7]))
        mean_cortex_dice = np.append(mean_cortex_dice, np.mean(dice[3:5]))

        wrapped_target_label = model.wrap(target_label, backward_flow)
        dice = dice_score(torch.argmax(wrapped_target_label, dim=1),
                          torch.argmax(source_label, dim=0), num_classes=20)
        mean_dice = np.append(mean_dice, np.mean(dice))
        mean_white_matter_dice = np.append(mean_white_matter_dice, np.mean(dice[5:7]))
        mean_cortex_dice = np.append(mean_cortex_dice, np.mean(dice[3:5]))

    print(f"Mean Dice: {np.mean(mean_dice)}")
    print(f"Mean White Matter Dice: {np.mean(mean_white_matter_dice)}")
    print(f"Mean Cortex Dice: {np.mean(mean_cortex_dice)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Registration Test 3D Images')
    parser.add_argument('-p', '--csv_path', help='csv file ', type=str, required=False, default="./dataset.csv")
    parser.add_argument('--load', help='Input model', type=str, required=False, default="./model.pth")
    parser.add_argument('--precision', help='Precision for Lightning trainer (16, 32 or 64)', type=int, required=False,
                        default=32)
    main(parser.parse_args())
    print("Success!")
