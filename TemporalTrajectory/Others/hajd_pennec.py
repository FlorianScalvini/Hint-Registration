import os
import csv
import monai
import torch
import argparse
import itertools
import torchio2 as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from dataset import subjects_from_csv
from Registration import RegistrationModuleSVF
import torchvision.transforms.functional as TF
from utils import get_cuda_is_available_or_cpu, create_directory, map_labels_to_colors, seg_map_error, write_namespace_arguments


NUM_CLASSES = 20

def test(args):
    # Config Dataset / Dataloader

    device = get_cuda_is_available_or_cpu()
    loggers = pl.loggers.TensorBoardLogger(save_dir= "./log" , name=None)
    save_path = loggers.log_dir.replace('log', "Results")
    create_directory(save_path)
    write_namespace_arguments(args, log_file=os.path.join(save_path, "config.json"))

    transforms = tio.Compose([
        tio.transforms.RescaleIntensity(out_min_max=(0, 1)),
        tio.transforms.CropOrPad(target_shape=221),
        tio.Resize(args.inshape),
        tio.OneHot(args.num_classes)
    ])

    subject_t0 = None
    subjects_list = subjects_from_csv(dataset_path=args.csv, lambda_age=lambda x: (x -args.t0) / (args.t1 - args.t0))
    subjects_dataset_transformed = tio.SubjectsDataset(subjects_list, transform=transforms)
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    reverse_transform = tio.Compose([
        tio.Resize(221),
        tio.CropOrPad(target_shape=subjects_dataset[0]['image'][tio.DATA].shape[1:]),
        tio.OneHot(args.num_classes)
    ])

    transformationPairs = list(itertools.combinations(range(len(subjects_dataset_transformed)), 2))
    in_shape = subjects_dataset_transformed[0]['image'][tio.DATA].shape[1:]

    # Load models
    reg_model = RegistrationModuleSVF(model=monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=2, out_channels=3, channels=[4, 8, 16, 32], strides=[2,2,2]), inshape=in_shape, int_steps=7)
    reg_model.load_state_dict(torch.load(args.load))
    reg_model.eval().to(device)

    denum = 0
    num = torch.zeros([1, 3] + list(in_shape)).to(device)

    with torch.no_grad():
        for i, j in transformationPairs:
            sample_i = subjects_dataset_transformed[i]
            sample_j = subjects_dataset_transformed[j]
            time_ij = sample_j['age'] - sample_i['age']
            velocity_ij = reg_model(sample_i['image'][tio.DATA].float().unsqueeze(dim=0).to(device),
                                    sample_j['image'][tio.DATA].unsqueeze(dim=0).float().to(device))
            num += velocity_ij * time_ij
            denum += time_ij * time_ij
        weightedMeanVelocity = num / denum if denum != 0 else torch.zeros_like(num)


    source = None
    for s in subjects_dataset_transformed:
        if s.age == 0:
            source = s
            break

    source_image = torch.unsqueeze(source["image"][tio.DATA], 0).to(device)
    source_label = torch.unsqueeze(source["label"][tio.DATA], 0).to(device)


    dice_metric = DiceMetric(include_background=True, reduction="none")

    with open(save_path + "/results.csv", mode='w') as file:
        header = ["time", "mDice", "Cortex", "Ventricule", "all"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for i in range(len(subjects_dataset_transformed)):
            target_subject = subjects_dataset_transformed[i]
            age = int(target_subject['age'] * (args.t1 - args.t0) + args.t0)
            weighted_velocity = weightedMeanVelocity * (target_subject['age'] - source['age'])
            forward_flow, backward_flow = reg_model.velocity_to_flow(velocity=weighted_velocity)
            warped_source_label = reg_model.warp(source_label.float(), forward_flow)
            warped_source_image = reg_model.warp(source_image.float(), forward_flow)

            warped_subject = tio.Subject(
                image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0),
                                      affine=source['image'].affine),
                label=tio.LabelMap(tensor=torch.argmax(torch.round(warped_source_label), dim=1).int().detach().cpu(),
                                   affine=source['label'].affine)
            )
            warped_subject = reverse_transform(warped_subject)

            if args.save_image:
                warped_subject['image'].save(save_path + "/" + str(age) + "_fused_source_image.nii.gz")
                warped_subject['label'].save(save_path + "/" + str(age) + "_fused_source_label.nii.gz")
            if args.error_map:
                colored_error_seg = map_labels_to_colors(
                    seg_map_error(warped_subject["label"][tio.DATA].unsqueeze(0),
                                  warped_subject["label"][tio.DATA].unsqueeze(0),
                                  dim=1)).squeeze().permute(3, 0, 1, 2)

                loggers.experiment.add_image("Atlas Sagittal Plane",
                                             TF.rotate(colored_error_seg[:, int(in_shape[0] / 2), :, :], 90), age)
                loggers.experiment.add_image("Atlas Coronal Plane",
                                             TF.rotate(colored_error_seg[:, :, int(in_shape[1] / 2), :], 90), age)
                loggers.experiment.add_image("Atlas Axial Plane",
                                             TF.rotate(colored_error_seg[:, :, :, int(in_shape[2] / 2)], 90), age)
            dice = dice_metric(warped_subject['label'][tio.DATA].unsqueeze(0).to(device), subjects_dataset[i]["label"][tio.DATA].unsqueeze(0).float().to(device))

            writer.writerow({
                "time": age,
                "mDice": torch.mean(dice[0][1:]).item(),
                "Cortex": torch.mean(dice[0][3:5]).item(),
                "Ventricule": torch.mean(dice[0][7:9]).item(),
                "all": dice[0].cpu().numpy()
            })
            loggers.experiment.add_scalar("Dice ventricule", torch.mean(dice[0][7:9]).item(), age)
            loggers.experiment.add_scalar("Dice cortex", torch.mean(dice[0][3:5]).item(), age)
            loggers.experiment.add_scalar("mDice", torch.mean(dice[0]).item(), age)



if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo Registration 3D Longitudinal Images with Pennec approach')
    parser.add_argument('--csv', type=str, help='Path to the csv file', default='../../data/full_dataset.csv')
    parser.add_argument('--t0', type=int, help='Initial time point', default=21)
    parser.add_argument('--t1', type=int, help='Final time point', default=36)
    parser.add_argument('--load', type=str, help='Path to the model', default='/home/florian/Documents/Dataset/JeanZay/Registration/Results/version_0/last_model.pth')
    parser.add_argument('--inshape', type=int, help='Size of the input image', default=128)
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--error_map', type=bool, help='Compute the error map', default=False)
    parser.add_argument('--save_image', type=bool, help='Save MRI', default=False)
    args = parser.parse_args()
    test(args=args)
