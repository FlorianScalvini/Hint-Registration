import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    '''
        Multi-layer perceptron
    '''
    def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_dim=None):
        '''
        :param input_dim: int
        :param output_dim: int
        :param hidden_dim: List of int
        '''
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = [32, 32, 32]
        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.ReLU()]
        for i in range(len(hidden_dim) - 2):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim[-2], output_dim))
        layers.append(nn.Hardtanh(min_val=-1.0, max_val=1.0))
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