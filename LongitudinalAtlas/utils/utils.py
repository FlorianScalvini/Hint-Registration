import numpy as np

def normalize_to_0_1(volume):
    """
        Normalize the image to (0,1) to be viewed as a greyscale
    Args:
        volume: 4D tensor (B, C, H, W)

    Returns:
        Normalized volume
    """
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)


palette = np.array([
    [0, 0, 0],  # 0:
    [244, 35, 232],  # 1:
    [70, 70, 70],  # 2:
    [102, 102, 156],  # 3:
    [190, 153, 153],  # 4:
    [153, 153, 153],  # 5:
    [250, 170, 30],  # 6:
    [220, 220, 0],  # 7:
    [0, 128, 0],  # 8:
    [0, 255, 0],  # 9:
    [128, 128, 0],  # 10:
    [128, 0, 0],  # 11:
    [255, 0, 0],  # 12:
    [0, 0, 128],  # 13:
    [0, 0, 255],  # 14:
    [0, 128, 128],  # 15:
    [255, 255, 0],  # 16:
    [255, 0, 255],  # 17:
    [192, 192, 192],  # 18:
    [255, 255, 255],  # 19:
])


# Create a colored segmentation map
def color_segmentation_map(segmentation_map):
    height, width = segmentation_map.shape
    colored_map = np.zeros((3, width, height), dtype=np.uint8)
    for label in range(palette.shape[0]):
        colored_map[segmentation_map == label] = palette[label]
    return colored_map