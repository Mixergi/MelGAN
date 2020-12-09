import torch.nn as nn
from torch.nn.utils import weight_norm


def WMConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WMConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))
