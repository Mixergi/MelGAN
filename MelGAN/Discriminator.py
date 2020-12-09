import torch.nn as nn

from melgan.weight_layer import WMConv1d


class DiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        down_sampling = 4

        channels = [64, 256, 1024, 1024]

        prev_channel = 16

        self.blocks = nn.ModuleList([nn.Sequential(
            nn.ReflectionPad1d(7),
            WMConv1d(1, prev_channel, 15),
            nn.LeakyReLU(0.3)
        )])
        
        for i, channel in enumerate(channels):
            self.blocks.extend([
                nn.Sequential(
                    nn.ReflectionPad1d(20),
                    WMConv1d(prev_channel, channel, 41, 4, groups=4**(i+1)),
                    nn.LeakyReLU(0.3)
                )
            ])

            prev_channel = channel

        self.blocks.extend([
            nn.Sequential(
                nn.ReflectionPad1d(2),
                WMConv1d(prev_channel, 1024, 5, 1),
                nn.LeakyReLU(0.3)
            ), nn.Sequential(
                nn.ReflectionPad1d(1),
                WMConv1d(1024, 1, 3, 1)
            )
        ])

    def forward(self, x):
        outputs = []

        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        return outputs  


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            DiscriminatorBlock() for _ in range(3)
        ])

        self.avg = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):

        outputs = []

        for block in self.blocks:
            outputs.append(block(x))
            x = self.avg(x)

        return outputs