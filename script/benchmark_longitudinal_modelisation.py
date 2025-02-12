import csv
import sys
import torch
import monai
import argparse
import numpy as np
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(1, "..")
import torchio2 as tio
from dataset import subjects_from_csv
from Registration import  RegistrationModuleSVF
from TemporalTrajectory.LongitudinalDeformation import HadjHamouLongitudinalDeformation, OurLongitudinalDeformation


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
        reg_model=RegistrationModuleSVF(
            int_steps=7,
            model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8, 16, 32),
                                                    strides=(2, 2)), inshape=(rshape, rshape, rshape))
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
    model = HadjHamouLongitudinalDeformation(
        t0=t0,
        t1=t1,
        device=device,
        reg_model=RegistrationModuleSVF(
            int_steps=7,
            model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8, 16, 32),
                                                    strides=(2, 2)), inshape=(rshape, rshape, rshape))
    )
    model.loads(model_path).eval().to(device)

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
            forward_deformation, _ = model.getDeformationFieldFromTime(velocity, transformed_source_dataset[i]['age'])
            warped_source_label = model.reg_model.warp(source_transformed, forward_deformation)
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
    parser.add_argument('--our_mlp', type=bool, help='Our benchmark', required=False, default=True)
    parser.add_argument('--our_linear', type=bool, help='Our benchmark', required=False, default=True)
    parser.add_argument('--csv', type=str, help='Path to the target images/labels', required=False, default="E:/Postdoc/Temporal_trajectory/Hint-Registration/data/full_dataset.csv")
    parser.add_argument('--num_classes', type=int, help='Number of classes', required=False, default=20)
    parser.add_argument('--t0', type=str, help='T0', required=False, default=21)
    parser.add_argument('--t1', type=str, help='T0', required=False, default=36)
    parser.add_argument('--error_map', type=bool, help='Compute the error map', default=False)
    parser.add_argument('--save_image', type=bool, help='Save MRI', default=False)
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize size', default=None)
    parser.add_argument('--csize', type=int, nargs='+', help='Crop size', default=None)
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
    parser.add_argument('--load_hadj', type=str, help='Ours network model path', required=False, default="/home/florian/Documents/Programs/Hint-Registration/Registration/Results/version_39/last_model.pth")

    args = parser.parse_args()
    main(args)

