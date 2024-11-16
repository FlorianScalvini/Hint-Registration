
# Normalize the volume to 0-1 range
def normalize_to_0_1(volume):
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)

