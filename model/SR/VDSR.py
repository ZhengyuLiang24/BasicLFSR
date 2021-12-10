'''
@inproceedings{VDSR,
  title={Accurate image super-resolution using very deep convolutional networks},
  author={Kim, Jiwon and Lee, Jung Kwon and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1646--1654},
  year={2016}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from math import sqrt

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.scale_factor = args.scale_factor
        self.channels = 64

        self.input = nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, channels=self.channels, num_of_layer = 18)
        self.output = nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self, block, channels, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, Lr_Info):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False) # bicubic 插值
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


class Conv_ReLU_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, sqrt(2. / n))

    pass


class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss
