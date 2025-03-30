import torch
import torch.nn as nn
from .mlp import MLP


def positional_encoding_3d(positions, max_freq=10):
    encoded_positions = []
    freqs = 2 ** torch.arange(max_freq, dtype=torch.float32)
    for i in range(positions.shape[1]):
        pos = positions[:, i:i + 1]
        encoded = []
        for freq in freqs:
            encoded.append(torch.sin(pos * freq))
            encoded.append(torch.cos(pos * freq))
        encoded_positions.append(torch.cat(encoded, dim=-1))
    return torch.cat(encoded_positions, dim=-1)

class ImplicitNeuralNetwork(nn.Module):
    def __init__(self, size : list[int], hidden_dim=32, num_layers=4, max_freq=10):
        super(ImplicitNeuralNetwork, self).__init__()
        self.max_freq = max_freq
        self.size = size
        self.model = MLP(input_dim=(len(size) + 1) * self.max_freq * 2, output_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
        ranges = [torch.linspace(0, s - 1, s) for s in size]
        grids = torch.meshgrid(*ranges)
        coords = torch.stack([grid.flatten() for grid in grids], dim=-1)
        self.encoded_coords = positional_encoding_3d(coords, max_freq=self.max_freq)


    def forward(self, x):
        x = positional_encoding_3d(torch.full((self.encoded_coords.shape[0], 1), x), max_freq=self.max_freq)
        encoded_inputs = torch.cat([self.encoded_coords, x], dim=-1)  # Combine (x, y, z, t)
        y = self.model(encoded_inputs)
        y = y.view(self.size[0], self.size[1], self.size[2], -1).permute(3, 0, 1, 2)
        return y
