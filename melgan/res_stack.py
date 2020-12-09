import torch.nn as nn

from melgan.weight_layer import WMConv1d

class ResBlock(nn.Module):
    def __init__(self, input_size, kernel_size=3, dilatrion_rate=1):
        super().__init__()

        layers = [
            nn.LeakyReLU(0.3),
            nn.ReflectionPad1d(dilatrion_rate),
            WMConv1d(input_size, input_size, kernel_size,
                      dilation=dilatrion_rate),
            nn.LeakyReLU(0.3),
            nn.ReflectionPad1d(1),
            WMConv1d(input_size, input_size, kernel_size, dilation=1)
        ]

        self.block = nn.Sequential(*layers)

        self.shortcut = WMConv1d(input_size, input_size, 1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ResStack(nn.Module):
    def __init__(self, input_size, kernel_size=3, dilatrion_rate=1, m_conv=3, n_residual=3):
        super().__init__()

        layers = [ResBlock(input_size, kernel_size, int(dilatrion_rate*(m_conv**i)))
                  for i in range(n_residual)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
