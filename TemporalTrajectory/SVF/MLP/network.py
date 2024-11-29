import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, hidden_size, activation_layer=nn.ReLU, output_activation=None):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_size) - 2):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activation_layer())
        layers.append(nn.Linear(hidden_size[-2], 1))
        if output_activation is not None:
            layers.append(output_activation())
        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Initialize weights using Xavier (Glorot) initialization
                init.kaiming_normal(layer.weight)
                # Initialize biases to zeros
                if layer.bias is not None:
                    init.zeros_(layer.bias)