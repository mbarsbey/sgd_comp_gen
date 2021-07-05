# Two layer fully connected neural network

import torch
import torch.nn as nn
import copy

class MultiLayerNN(nn.Module):

    def __init__(self, input_dim=28*28, width=50, depth=2 , num_classes=10):
        assert depth >= 2
        super(MultiLayerNN, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.num_classes = num_classes
        
        layers = []
        layers.append(nn.Linear(self.input_dim, self.width, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth-2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.width, self.num_classes, bias=False))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    x = torch.randn(5, 1, 32, 32)
    net = MultiLayerNN(input_dim=32*32, width=123)
    print(net(x))
