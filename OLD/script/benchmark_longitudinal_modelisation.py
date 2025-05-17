import csv
import sys
import torch
import monai
import argparse
import numpy as np
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt
import torchio as tio
sys.path.insert(0, ".")
sys.path.insert(1, "..")

from src.utils.utils import subjects_from_csv
from src.modules.registration import  RegistrationModule
from src.modules.pairwise_registration import PairwiseRegistrationModuleVelocity
from src.modules.longitudinal_deformation import HadjHamouLongitudinalDeformation, OurLongitudinalDeformation
import torch.nn as nn
from torch import Tensor
from src.modules.blocks.spatial_transformation import SpatialTransformer, VecInt
class RegistrationModule(nn.Module):
    '''
        Registration module for 3D image registration with DVF
    '''
    def __init__(self, model: nn.Module, inshape: [int, int, int]):
        '''
        :param model: nn.Module
        :param inshape: [int, int, int]
        '''
        super().__init__()
        self.model = model
        self.spatial_transformer = SpatialTransformer(size=inshape)

    def forward(self, source: Tensor, target: Tensor) -> (Tensor, Tensor):
        '''
            Returns the deformation field from source to target image
        '''
        x = torch.cat([source, target], dim=1)
        y = self.model(x)
        return y

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor) -> (Tensor, Tensor):
        '''
        Returns the forward and backward flow
        '''
        return self.forward(source, target), self.forward(target, source)

    def warp(self, tensor: Tensor, flow: Tensor) -> Tensor:
        return self.spatial_transformer(tensor, flow)


    @staticmethod
    def regularizer(disp, penalty='l2') -> Tensor:
        '''
        Compute the regularizer of the displacement field
        '''
        dy = torch.abs(disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        dx = torch.abs(disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        dz = torch.abs(disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

class RegistrationModuleSVF(RegistrationModule):
    '''
        Registration module for 3D image registration with stationary velocity field
        based on the DVF Registration module
    '''
    def __init__(self, model: nn.Module, inshape: [int, int, int], int_steps: int = 7):
        '''
        :param model: nn.Module
        :param inshape: [int, int, int]
        :param int_steps: int
        '''
        super().__init__(model=model, inshape=inshape)
        self.vecint = VecInt(inshape=inshape, nsteps=int_steps) # Vector integration based on Runge-Kutta method

    def velocity_to_flow(self, velocity):
        '''
            Convert the velocity field to a flow field (forward and backward)
        '''
        forward_flow = self.vecint(velocity)
        backward_flow = self.vecint(-velocity)
        return forward_flow, backward_flow

    def load_network(self, path):
        '''
            Load the network from a file
        '''
        self.model.load_state_dict(torch.load(path))

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        '''
            Returns the forward and backward flow after registration
            Override the parent method with SVF intergration between the Registration Network and the DVF output
        '''
        velocity = self.forward(source, target)
        forward_flow, backward_flow = self.velocity_to_flow(velocity)
        return forward_flow, backward_flow


def _get_transforms(crshape: int | list[int], rshape: int | list[int], num_classes: int):
    return tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=crshape),
        tio.Resize(rshape),
        tio.OneHot(num_classes)
    ])

def _get_reverse_transform(crshape: int | list[int], rshape: int | list[int], num_classes: int):
    return tio.Compose([
        tio.Resize(rshape),
        tio.CropOrPad(target_shape=crshape),
        tio.OneHot(num_classes)
    ])


def _compute_ours(dataset_path: str, model_path: str, crshape: list[int] | int, rshape: list[int] | int, num_classes: int,
                  t0: int, t1: int, device: str, mode: str, model_mlp: str | None = None,
                  mlp_num_layers: str | None = None, mlp_hidden_dim: int | None = None):
    model = OurLongitudinalDeformation(
        t0=t0,
        t1=t1,
        mode=mode,
        num_layers=mlp_num_layers,
        hidden_dim=mlp_hidden_dim,
        reg_model=PairwiseRegistrationModuleVelocity(
            int_steps=7,
            model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3,
                                                    channels=(8, 16, 32),strides=(2, 2)))
    )

    if mode == 'mlp' and model_mlp is not None:
        model.loads(model_path, model_mlp)
    else:
        model.loads(model_path)
    model.eval().to(device)

    transforms = _get_transforms(crshape=crshape, rshape=rshape, num_classes=num_classes)
    subjects_list = subjects_from_csv(dataset_path=dataset_path,
                                      lambda_age=lambda x: (x - t0) / (t1 - t0))
    source_dataset = tio.SubjectsDataset(subjects_list, transform=None)
    reverse_transform = _get_reverse_transform(rshape=crshape, crshape=source_dataset[0]["image"][tio.DATA].shape[1:],
                                               num_classes=num_classes)

    transformed_source_dataset = tio.SubjectsDataset(subjects_list, transform=transforms)
    dice_scores = []
    dice_metric = DiceMetric(include_background=True, reduction="none")
    source_idx = -1
    target_idx = -1

    for i in range(len(source_dataset)):
        if source_dataset[i]['age'] == 0:
            source_idx = i
        elif source_dataset[i]['age'] == 1:
            target_idx = i
    model.eval()
    with torch.no_grad():
        velocity = model.forward((transformed_source_dataset[source_idx]['image'][tio.DATA].unsqueeze(0).to(device),
                                  transformed_source_dataset[target_idx]['image'][tio.DATA].unsqueeze(0).to(device)))
        source_transformed = transformed_source_dataset[source_idx]['label'][tio.DATA].unsqueeze(0).float().to(device)
        for i in range(len(transformed_source_dataset)):
            forward_deformation, _ = model.getDeformationFieldFromTime(velocity, transformed_source_dataset[i]['age'])
            warped_source_label = model.reg_model.warp(source_transformed, forward_deformation)
            warped_source_label = reverse_transform(warped_source_label.squeeze(0).cpu()).unsqueeze(0).to(device)
            target_label = source_dataset[i]["label"][tio.DATA].float().to(device).unsqueeze(0)
            dice = dice_metric(warped_source_label, target_label)[0]
            dice_metric.reset()
            dice_scores.append(dice.cpu().numpy())
    return np.vstack(dice_scores)


def _compute_hadj(dataset_path: str, model_path: str, crshape: int | list[int], rshape: int | list[int],
                  num_classes: int, t0: int, t1: int, device: str):
    reg_model = RegistrationModuleSVF(
            int_steps=7,
            model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8, 16, 32),
                                                    strides=(2, 2)),
            inshape=rshape
    )
    reg_model.load_state_dict(torch.load(model_path), strict=False)
    model = HadjHamouLongitudinalDeformation(
        t0=t0,
        t1=t1,
        reg_model=reg_model.eval().to(device))

    transforms = _get_transforms(crshape=crshape, rshape=rshape, num_classes=num_classes)
    subjects_list = subjects_from_csv(dataset_path=dataset_path, lambda_age=lambda x: (x -t0) / (t1 - t0))
    source_dataset = tio.SubjectsDataset(subjects_list, transform=None)
    reverse_transform = _get_reverse_transform(rshape=crshape, crshape=source_dataset[0]["image"][tio.DATA].shape[1:], num_classes=num_classes)
    transformed_source_dataset = tio.SubjectsDataset(subjects_list, transform=transforms)

    dice_scores = []
    dice_metric = DiceMetric(include_background=True, reduction="none")

    idx = -1
    for i in range(len(source_dataset)):
        if source_dataset[i]['age'] == 0:
            idx = i
    model.eval()
    with torch.no_grad():
        velocity = model.forward(transformed_source_dataset)
        source_transformed = transformed_source_dataset[idx]['label'][tio.DATA].unsqueeze(0).float().to(device)
        for i in range(len(transformed_source_dataset)):
            forward_deformation = velocity * transformed_source_dataset[i]['age']
            warped_source_label = model.model.warp(source_transformed, forward_deformation)
            warped_source_label = reverse_transform(warped_source_label.squeeze(0).cpu()).unsqueeze(0).to(device)
            target_label = source_dataset[i]["label"][tio.DATA].float().to(device).unsqueeze(0)
            dice = dice_metric(warped_source_label, target_label)[0]
            dice_metric.reset()
            dice_scores.append(dice.cpu().numpy())
    return np.vstack(dice_scores)


def _compute_ants(dataset_path, num_classes):
    return NotImplemented


def main(args):

    scores = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.our_mlp:
        scores["Ours-mlp"] = _compute_ours(dataset_path=args.csv, model_path=args.load_reg_mlp, crshape=args.csize,
                                           rshape=args.rsize, num_classes=args.num_classes, t0=args.t0,
                                           t1=args.t1, device=device, mode='mlp', model_mlp=args.load_mlp,
                                           mlp_hidden_dim=args.ml_hidden_dim, mlp_num_layers=args.mlp_num_layers)

    if args.our_linear:
        scores["Ours-linear"] = _compute_ours(dataset_path=args.csv, model_path=args.load_reg_linear,
                                              rshape=args.csize, crshape=args.rsize, num_classes=args.num_classes,
                                              t0=args.t0, t1=args.t1, device=device, mode='linear')

    if args.hadj_hamou:
        scores["Hadj-Hamou"] = _compute_hadj(dataset_path=args.csv, model_path=args.load_hadj, crshape=args.csize,
                                             rshape=args.rsize, num_classes=args.num_classes, t0=args.t0, t1=args.t1,
                                             device=device)
    with open("./results_long.csv", mode='w') as file:
        header = ["index", "Ours-mlp", "Ours-linear", "Hadj-Hamou"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for i in range(len(scores[next(iter(scores))])):
            save = {"index": i}
            for key in scores.keys():
                save[key] = scores[key][i]
            writer.writerow(save)

    color = ["magenta", "peru", "green"]
    marker = ["D", "h", "x"]

    x = [i for i in range(args.t0, args.t0 + len(scores[next(iter(scores))]))]
    for i in range(len(scores)):
        label = list(scores.keys())[i]
        plt.plot(x,  np.mean(scores[label][:, 1:], axis=1).tolist(), color=color[i], marker=marker[i], label=label)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')

    parser.add_argument('--hadj_hamou', type=bool, help='Hadj-Hamou benchmark', required=False, default=True)
    parser.add_argument('--our_mlp', type=bool, help='Our benchmark', required=False, default=False)
    parser.add_argument('--our_linear', type=bool, help='Our benchmark', required=False, default=False)
    parser.add_argument('--csv', type=str, help='Path to the target images/labels', required=False, default="/home/florian/Documents/Programs/Hint-Registration/dataset/dHCP/dataset.csv")
    parser.add_argument('--num_classes', type=int, help='Number of classes', required=False, default=20)
    parser.add_argument('--t0', type=str, help='T0', required=False, default=21)
    parser.add_argument('--t1', type=str, help='T0', required=False, default=36)
    parser.add_argument('--error_map', type=bool, help='Compute the error map', default=False)
    parser.add_argument('--save_image', type=bool, help='Save MRI', default=False)
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize size', default=[192,224,192])
    parser.add_argument('--csize', type=int, nargs='+', help='Crop size', default=[192,224,192])
    # linear method
    parser.add_argument('--load_reg_linear', type=str, help='Ours network model path', required=False, default="/home/florian/Documents/Programs/Hint-Registration/Registration/Results/version_39/last_model.pth")
    # MLP Method
    parser.add_argument('--load_reg_mlp', type=str, help='Ours network model path', required=False,
                        default="/home/florian/Documents/Programs/Hint-Registration/Registration/Results/version_39/last_model.pth")
    parser.add_argument('--load_mlp', type=str, help='Ours network model path', required=False,
                        default="/home/florian/Documents/Programs/Hint-Registration/Registration/Results/version_39/last_model.pth")
    parser.add_argument('--mlp_num_layers', type=int, help='Number of layer of the mlp', default=4)
    parser.add_argument('--mlp_hidden_dim', type=int, help='Number of layer of the mlp', default=32)
    #Hadj_Hamou method
    parser.add_argument('--load_hadj', type=str, help='Ours network model path', required=False, default="/home/florian/JeanZay/regis/Results/version_22/last_model.pth")

    args = parser.parse_args()
    main(args)

