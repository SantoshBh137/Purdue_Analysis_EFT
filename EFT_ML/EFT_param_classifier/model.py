import torch.nn as nn

class ParametricClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        for _ in range(3):
            layers += [nn.Linear(input_dim,100), nn.ReLU(), nn.BatchNorm1d(100), nn.Dropout(0.1)]
            input_dim = 100
        layers += [nn.Linear(100,1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)
