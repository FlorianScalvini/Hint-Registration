import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    '''
        Multi-layer perceptron
    '''
    def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_dim: int = 32, num_layers: int = 4):
        '''
        :param input_dim: int
        :param output_dim: int
        :param hidden_dim: int
        :param num_layers: int
        '''
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))  # First layer : Coord + time
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
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