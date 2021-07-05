import math
import torch.nn as nn
import torch.nn.init as init

__all__ = ['VGG', 'vgg11', 'vgg11_bw']

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 10, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, bw=False):
    layers = []
    if bw:
        in_channels = 1
    else:
        in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'F': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] # added for mnist
}


def vgg11():
    return VGG(make_layers(cfgs['A']))

def vgg11_bw():
    return VGG(make_layers(cfgs['F'], bw=True))