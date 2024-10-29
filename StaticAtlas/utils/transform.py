import torch


def normalize_to_0_1(volume):
    shape = volume.shape
    volume = volume.view(volume.size(0), -1)
    volume -= volume.min(1, keepdim=True)[0]
    volume /= volume.max(1, keepdim=True)[0]
    volume = volume.view(shape)
    return volume
