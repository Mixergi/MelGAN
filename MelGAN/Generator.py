import torch.nn as nn

from melgan.res_stack import ResStack
from melgan.weight_layer import WMConv1d, WMConvTranspose1d


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 512

        upsample = [8, 8, 2, 2]

        layers = [
            nn.ReflectionPad1d(3),
            WMConv1d(80, input_size, 7)
        ]

        for i, s in enumerate(upsample):
            input_size //= 2
            
            layers += [
                nn.LeakyReLU(0.3),
                WMConvTranspose1d(
                    input_size*2, input_size, s*2, s, padding=s//2+s%2),
                ResStack(input_size)
            ]

        layers += [
            nn.LeakyReLU(0.3),
            nn.ReflectionPad1d(3),
            WMConv1d(input_size, 1, 7, 1),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
