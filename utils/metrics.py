import torch
from torch import Tensor
from torchmetrics import F1Score
import logging
import torchio as tio
import pandas as pd
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

class SegmentationMetric():
    def __init__(self, labels : dict[int,str], list_class : list[int]):
        self.metric = F1Score(task="multiclass", num_classes=len(labels), average=None)
        self.list_class = list_class
        self.labels = labels

    def update(self, preds: Tensor, targets: Tensor) -> None:
        self.metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        data = self.metric.compute()
        metric = {labels[idx]: data[idx].numpy().item() for idx in list_class}
        metric['average'] = torch.mean(data).numpy().item()
        return metric

    def __str__(self):
        result = [f"{k} : {v}" for k, v in self.compute().items()]
        # Join the list into a single string with newline characters
        return "\n".join(result)

    def reset(self) -> None:
        self.metric.reset()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = False,
                        default="/media/ubuntu/SSD Samsung/Postdoc/Dataset/template_dHCP/fetal_brain_mri_atlas/parcellations/tissue-t23.00_dhcp-19.nii.gz")
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = False,
                        default="/home/ubuntu/Documents/Result/Biblio/ANTS/ants_warped_target_label.nii.gz")
    parser.add_argument( '--list_class', help='Number of classes', type=str, required = False,
                        default="3,4,5,6")
    parser.add_argument( '--tsv_labelname', help='List of label names', type=str, required = False,
                        default="/media/ubuntu/SSD Samsung/Postdoc/Dataset/template_dHCP/fetal_brain_mri_atlas/info/dhcp-atlas-summary-info-19-labels.csv")

    args = parser.parse_args()


    subject_target = tio.Subject(
        label=tio.LabelMap(args.target),
    )

    subject_source = tio.Subject(
        label=tio.LabelMap(args.source),
    )

    transform = tio.Compose(
        [
            tio.CropOrPad(192),
            tio.Resize(175)
        ]
    )

    #subject_target = transform(subject_target)


    df = pd.read_csv(args.tsv_labelname, sep=',')
    labels = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    list_class = []
    try:
        list_class = list(map(int, args.list_class.split(',')))
    except ValueError:
        logging.exception("An exception was thrown!")
    metric = SegmentationMetric(labels=labels, list_class=list_class)
    metric.update(subject_source["label"][tio.DATA], subject_target["label"][tio.DATA])
    print(metric)



def multi_class_score(one_class_fn, predictions, labels, num_classes, one_hot=False):
    result = np.zeros(num_classes)
    for label_index in range(num_classes):
        if one_hot:
            class_predictions = predictions[:, label_index, ...]
            class_labels = labels[:, label_index, ...]
        else:
            class_predictions = predictions.eq(label_index)
            class_predictions = class_predictions.squeeze(1)  # remove channel dim
            class_labels = labels.eq(label_index)
            class_labels = class_labels.squeeze(1)  # remove channel dim
        class_predictions = class_predictions.float()
        class_labels = class_labels.float()


        result[label_index] = one_class_fn(class_predictions, class_labels).mean()

    return result


def hausdorff_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    """
    Calc  hausdorff distance

    Args:
        predictions ([torch tensor]): batch of images [NDHW]
        labels ([torch tensor]): batch of image [NDHW]
        num_classes ([int]): [number of segmetation classes]
        spacing (list, float): [voxel spacings]. Defaults to [1, 1, 1].
    return   dict: ['label'] = [B, score]
    """
    def one_class_hausdorff_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        try:
            pred_cpu_array = pred.cpu().numpy()
            label_cpu_array = lab.cpu().numpy()
        except:
            print ('tensors are not in CUDA memory')
            pred_cpu_array = pred.numpy()
            label_cpu_array = lab.numpy()

        for i in range(batch):
            pred_array = pred_cpu_array[i]
            target_array = label_cpu_array[i]

            pred_img = sitk.GetImageFromArray(pred_array)
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(target_array)
            lab_img.SetSpacing(spacing)
            try:
                hausdorff_distance_filter.Execute(pred_img, lab_img)
                hd = hausdorff_distance_filter.GetHausdorffDistance()
            except:
                hd = np.inf
            result.append(hd)
        return torch.tensor(np.asarray(result))
    return multi_class_score(one_class_hausdorff_distance, predictions, labels, num_classes=num_classes)


def average_surface_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    def one_class_average_surface_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        try:
            pred_cpu_array = pred.cpu().numpy()
            label_cpu_array = lab.cpu().numpy()
        except:
            print ('tensors are not in CUDA memory')
            pred_cpu_array = pred.numpy()
            label_cpu_array = lab.numpy()

        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred_cpu_array[i])
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(label_cpu_array[i])
            lab_img.SetSpacing(spacing)
            try:
                hausdorff_distance_filter.Execute(pred_img, lab_img)
                hd = hausdorff_distance_filter.GetAverageHausdorffDistance()
            except:
                hd = np.inf
            result.append(hd)
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_average_surface_distance, predictions, labels, num_classes=num_classes)



def dice_score(prediction, labels):
    # Step 4: Compute intersection and union for each class
    intersection = np.sum(prediction * labels, axis=(1, 2, 3))  # Shape: [nb_classes]
    union = np.sum(prediction, axis=(1, 2, 3)) + np.sum(labels, axis=(1, 2, 3))  # Shape: [nb_classes]
    # Step 5: Compute Dice score for each class
    try:
        dice = 2 * intersection / (union)
    except ZeroDivisionError:
        dice = 2 * intersection / (union) + 1e-6 # Avoid division by zero
    return dice


def dice_score_old(predictions, labels, num_classes):
    """ returns the dice score

    Args:
        predictions: one hot tensor [B, num_classes, D, H, W]
        labels: label tensor [B, 1, D, H, W]
    Returns:
        dict: ['label'] = [B, score]
    """

    def one_class_dice(pred, lab):
        shape = pred.shape
        true_positive = (pred * lab).sum()
        try:
            dc = (2. * true_positive) / (pred.sum() + lab.sum())
        except ZeroDivisionError:
            ## NaN value ref: https://github.com/deepmind/surface-distance,
            dc = torch.tensor(np.NaN)
        return dc
    return multi_class_score(one_class_dice, predictions, labels, num_classes=num_classes)


def precision(predictions, labels, num_classes):
    def one_class_precision(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        try:
            value =  true_positive / p_flat.sum()
        except:
            value = torch.tensor(np.NaN)
        return value

    return multi_class_score(one_class_precision, predictions, labels, num_classes=num_classes)


def recall(predictions, labels, num_classes):
    def one_class_recall(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        negative = 1 - p_flat
        false_negative = (negative * l_flat).sum()
        try:
            value = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            value= torch.tensor(np.NaN)
        return value
    return multi_class_score(one_class_recall, predictions, labels, num_classes=num_classes)


class Logger():
    def __init__(self, name, loss_names):
        self.name = name
        self.loss_names = loss_names
        self.epoch_logger = {}
        self.epoch_summary = {}
        self.epoch_number_logger = []
        self.reset_epoch_logger()
        self.reset_epoch_summary()

    def reset_epoch_logger(self):
        for loss_name in self.loss_names:
            self.epoch_logger[loss_name] = []

    def reset_epoch_summary(self):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name] = []

    def update_epoch_logger(self, loss_dict):
        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                self.epoch_logger[loss_name].append(loss_value)

    def update_epoch_summary(self, epoch, reset=True):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name].append(np.mean(self.epoch_logger[loss_name]))
        self.epoch_number_logger.append(epoch)
        if reset:
            self.reset_epoch_logger()

    def get_latest_dict(self):
        latest = {}
        for loss_name in self.loss_names:
            latest[loss_name] = self.epoch_summary[loss_name][-1]
        return latest

    def get_epoch_logger(self):
        return self.epoch_logger

    def write_epoch_logger(self, location, index, loss_names, loss_labels, colours, linestyles=None, scales=None,
                           clear_plot=True):
        if linestyles is None:
            linestyles = ['-'] * len(colours)
        if scales is None:
            scales = [1] * len(colours)
        if not (len(loss_names) == len(loss_labels) and len(loss_labels) == len(colours) and len(colours) == len(
                linestyles) and len(linestyles) == len(scales)):
            raise ValueError('Length of all arg lists must be equal but got {} {} {} {} {}'.format(len(loss_names),
                                                                                                   len(loss_labels),
                                                                                                   len(colours),
                                                                                                   len(linestyles),
                                                                                                   len(scales)))

        for name, label, colour, linestyle, scale in zip(loss_names, loss_labels, colours, linestyles, scales):
            if scale == 1:
                plt.plot(range(0, len(self.epoch_logger[name])), self.epoch_logger[name], c=colour,
                         label=label, linestyle=linestyle)
            else:
                plt.plot(range(0, len(self.epoch_logger[name])), [scale * val for val in self.epoch_logger[name]],
                         c=colour,
                         label='{} x {}'.format(scale, label), linestyle=linestyle)
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('{}/{}.png'.format(location, index))
        if clear_plot:
            plt.clf()

    def print_latest(self, loss_names=None):
        print_str = '{}\tEpoch: {}\t'.format(self.name, self.epoch_number_logger[-1])
        if loss_names is None:
            loss_names = self.loss_names
        for loss_name in loss_names:
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                print_str += '{}: {:.6f}\t'.format(loss_name, self.epoch_summary[loss_name][-1])
        print(print_str)