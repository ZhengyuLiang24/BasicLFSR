'''
@article{LF-DFnet,
  author  = {Wang, Yingqian and Yang, Jungang and Wang, Longguang and Ying, Xinyi and Wu, Tianhao and An, Wei and Guo, Yulan},
  title   = {Light Field Image Super-Resolution Using Deformable Convolution},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30),
  pages   = {1057-1071},
  year    = {2021},
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
from math import sqrt
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from dcn.modules.deform_conv import DeformConv


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        n_blocks, channel = 4, 32
        self.factor = factor
        self.angRes = angRes
        self.FeaExtract = FeaExtract(channel)
        self.ADAM_1 = ADAM(channel, angRes)
        self.ADAM_2 = ADAM(channel, angRes)
        self.ADAM_3 = ADAM(channel, angRes)
        self.Reconstruct = CascadedBlocks(n_blocks, 4*channel)
        self.UpSample = Upsample(channel, factor)


    def forward(self, x):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
        x_sv, x_cv = LFsplit(x, self.angRes)
        buffer_sv_0, buffer_cv_0 = self.FeaExtract(x_sv, x_cv)
        buffer_sv_1, buffer_cv_1 = self.ADAM_1(buffer_sv_0, buffer_cv_0)
        buffer_sv_2, buffer_cv_2 = self.ADAM_2(buffer_sv_1, buffer_cv_1)
        buffer_sv_3, buffer_cv_3 = self.ADAM_3(buffer_sv_2, buffer_cv_2)
        buffer_sv = torch.cat((buffer_sv_0, buffer_sv_1, buffer_sv_2, buffer_sv_3), 2)
        buffer_cv = torch.cat((buffer_cv_0, buffer_cv_1, buffer_cv_2, buffer_cv_3), 1)

        buffer_sv = self.Reconstruct(buffer_sv)
        out_sv = self.UpSample(buffer_sv)

        buffer_cv = self.Reconstruct(buffer_cv.unsqueeze(1))
        out_cv = self.UpSample(buffer_cv)

        out = FormOutput(out_sv, out_cv) + x_upscale

        return out


class Upsample(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(4*channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out


class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x_sv, x_cv):
        buffer_cv_0 = self.FEconv(x_cv)
        buffer_cv = self.FERB_1(buffer_cv_0)
        buffer_cv = self.FERB_2(buffer_cv)
        buffer_cv = self.FERB_3(buffer_cv)
        buffer_cv = self.FERB_4(buffer_cv)

        b, n, h, w = x_sv.shape
        x_sv = x_sv.contiguous().view(b*n, -1, h, w)
        buffer_sv_0 = self.FEconv(x_sv)
        buffer_sv = self.FERB_1(buffer_sv_0)
        buffer_sv = self.FERB_2(buffer_sv)
        buffer_sv = self.FERB_3(buffer_sv)
        buffer_sv = self.FERB_4(buffer_sv)
        _, c, h, w = buffer_sv.shape
        buffer_sv = buffer_sv.unsqueeze(1).contiguous().view(b, -1, c, h, w)  # buffer_sv:  B, N, C, H, W

        return buffer_sv, buffer_cv


class ADAM(nn.Module):
    def __init__(self, channel, angRes):
        super(ADAM, self).__init__()
        self.conv_1 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.ASPP = ResASPP(channel)
        self.conv_off = nn.Conv2d(channel, 2*9, kernel_size=1, stride=1, padding=0)
        self.conv_off.lr_mult = 0.1
        self.init_offset()
        self.conv_f1 = nn.Conv2d(angRes*angRes*channel, angRes*angRes*channel, kernel_size=1, stride=1, padding=0)

        self.conv_f3 = nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0)
        self.dcn = DeformConv(channel, channel, 3, 1, 1, deformable_groups=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def init_offset(self):
        self.conv_off.weight.data.zero_()
        self.conv_off.bias.data.zero_()

    def forward(self, x_sv, x_cv):
        b, n, c, h, w = x_sv.shape
        aligned_fea = []
        for i in range(n):
            current_sv = x_sv[:, i, :, :, :].contiguous()
            buffer = torch.cat((current_sv, x_cv), dim=1)           # B * 2C * H * W
            buffer = self.lrelu(self.conv_1(buffer))
            buffer = self.ASPP(buffer)
            offset = self.conv_off(buffer)     # B*N, 18, H, W
            current_aligned_fea = self.lrelu(self.dcn(current_sv, offset)) # B*N, C, H, W
            aligned_fea.append(current_aligned_fea)
        aligned_fea = torch.cat(aligned_fea, dim=1)         # B, N*C, H, W
        fea_collect = torch.cat((aligned_fea, x_cv), 1)     # B, (N+1)*C, H, W
        fuse_fea = self.conv_f1(fea_collect)# B, (N+1)*C, H, W
        fuse_fea = fuse_fea.unsqueeze(1).contiguous().view(b, -1, c, h, w)  # B, N+1, C, H, W

        out_sv = []
        for i in range(n):
            current_sv = x_sv[:, i, :, :, :].contiguous()
            current_fuse = fuse_fea[:, i+1, :, :, :].contiguous()
            buffer = torch.cat((current_fuse, current_sv), dim=1)
            buffer = self.lrelu(self.conv_1(buffer))
            buffer = self.ASPP(buffer)
            offset = self.conv_off(buffer)  # B*N, 18, H, W
            dist_fea = self.lrelu(self.dcn(current_fuse, offset))
            fuse_sv = torch.cat((current_sv, dist_fea), dim=1)
            fuse_sv = self.conv_f3(fuse_sv)
            out_sv.append(fuse_sv)
        out_sv = torch.stack(out_sv, dim=1)
        out_cv = self.conv_f3(torch.cat((x_cv, fuse_fea[:, 0, :, :, :]), 1))
        return out_sv, out_cv


class CascadedBlocks(nn.Module):
    def __init__(self, n_blocks, channel):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(IMDB(channel))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.body[i](x)
        return x


class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class IMDB(nn.Module):
    def __init__(self, channel):
        super(IMDB, self).__init__()
        self.conv_0 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(3*channel//4, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(3*channel//4, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(3*channel//4, channel//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv_t = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer = x.contiguous().view(b*n, -1, h, w)
        buffer = self.lrelu(self.conv_0(buffer))
        buffer_1, buffer = ChannelSplit(buffer)
        buffer = self.lrelu(self.conv_1(buffer))
        buffer_2, buffer = ChannelSplit(buffer)
        buffer = self.lrelu(self.conv_2(buffer))
        buffer_3, buffer = ChannelSplit(buffer)
        buffer_4 = self.lrelu(self.conv_3(buffer))
        buffer = torch.cat((buffer_1, buffer_2, buffer_3, buffer_4), dim=1)
        buffer = self.lrelu(self.conv_t(buffer))
        x_buffer = buffer.contiguous().view(b, n, -1, h, w)
        return x_buffer + x


def ChannelSplit(input):
    _, C, _, _ = input.shape
    c = C//4
    output_1 = input[:, :c, :, :]
    output_2 = input[:, c:, :, :]
    return output_1, output_2


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel,channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            k = u*angRes + v
            if k != (angRes*angRes - 1)/2:
                data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])
            else:
                data_cv = data[:, :, u*h:(u+1)*h, v*w:(v+1)*w]
    data_sv = torch.cat(data_sv, dim=1)
    return data_sv, data_cv


def FormOutput(x_sv, x_cv):
    b, n, c, h, w = x_sv.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            k = u*angRes + v
            if k == n//2:
                buffer.append(x_cv[:, 0, :, :, :])
            else:
                buffer.append(x_sv[:, kk, :, :, :])
                kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out
