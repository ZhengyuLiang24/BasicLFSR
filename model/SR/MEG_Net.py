'''
@article{MEG_Net,
  title={End-to-end light field spatial super-resolution network using multiple epipolar geometry},
  author={Zhang, Shuo and Chang, Song and Lin, Youfang},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={5956--5968},
  year={2021},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import math

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        scale = args.scale_factor
        is_large_kernel = False
        n_seb = 4
        n_sab = 4
        n_feats = 32

        # 4 resblock in each image stack
        # n_seb = 4 #5
        if is_large_kernel:
            kernel_size = (3, 5, 5)
            padding = (1, 2, 2)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)

        # define body module
        m_horizontal_first = [
            nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_horizontal = [
            ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        m_vertical_first = [nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_vertical = [
            ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        m_45_first = [nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_45 = [
            ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        m_135_first = [nn.Conv3d(1, n_feats, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        m_135 = [
            ResBlockc3d(n_feats, is_large_kernel=is_large_kernel) for _ in range(n_seb)
        ]

        s_list = [ResBlock2d(4 * n_feats, 4 * n_feats, kernel_size=(1, 3, 3)) for _ in range(n_sab)]  # 4
        a_list = [ResBlock2d(4 * n_feats, 4 * n_feats, kernel_size=(1, 3, 3)) for _ in range(n_sab)]  # 4

        m_upsample = [
            nn.ConvTranspose3d(4 * n_feats, n_feats, kernel_size=(1, scale + 2, scale + 2), stride=(1, scale, scale),
                               # 4
                               padding=(0, 1, 1), output_padding=(0, 0, 0), bias=True),
            nn.Conv3d(n_feats, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)]

        m_upsample_main = [
            nn.ConvTranspose3d(1, 1, kernel_size=(1, scale + 2, scale + 2), stride=(1, scale, scale), padding=(0, 1, 1),
                               output_padding=(0, 0, 0), bias=True)]




        self.horizontal_first = nn.Sequential(*m_horizontal_first)
        self.horizontal = nn.Sequential(*m_horizontal)
        self.vertical_first = nn.Sequential(*m_vertical_first)
        self.vertical = nn.Sequential(*m_vertical)
        self.s45_first = nn.Sequential(*m_45_first)
        self.s45 = nn.Sequential(*m_45)
        self.s135_first = nn.Sequential(*m_135_first)
        self.s135 = nn.Sequential(*m_135)

        self.s_body_list = nn.ModuleList(s_list)
        self.a_body_list = nn.ModuleList(a_list)

        self.upsample = nn.Sequential(*m_upsample)
        self.upsample_main = nn.Sequential(*m_upsample_main)
        self.scale = scale
        self.n_feats = n_feats
        self.n_sab = n_sab





    def forward(self, lr, info=None):
        # reshape for LFSSR_TIP
        train_data = rearrange(lr, 'b c (a1 h) (a2 w) -> (b c) a1 a2 h w', a1=self.angRes, a2=self.angRes)

        # extract the central view from the image stack
        ''' super-resolution horizontally '''
        batch_size = (train_data.shape[0])
        view_n = (train_data.shape[1])
        image_h = int(train_data.shape[3])
        image_w = int(train_data.shape[4])  # train_data.shape[4]

        horizontal = torch.zeros((batch_size, self.n_feats, view_n, view_n,
                                  int(image_h), int(image_w)),
                                 dtype=torch.float32)
        horizontal = horizontal.to(lr.device)
        for i in range(0, view_n, 1):
            train_cut = train_data[:, i:i + 1, :, :, :]
            train_cut = self.horizontal_first(train_cut)
            train_cut = train_cut + self.horizontal(train_cut)  # (7,7,32,32)
            horizontal[:, :, i:i + 1, :, :, :] = train_cut.view(batch_size, self.n_feats, 1, view_n,
                                                                int(image_h), int(image_w))
        horizontal = horizontal.view(-1, self.n_feats, view_n * view_n, image_h, image_w)  # (1,49,64,64)

        ''' super-resolution vertically '''
        vertical = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)),
                               dtype=torch.float32)
        vertical = vertical.to(lr.device)
        for i in range(0, view_n, 1):
            train_cut = train_data[:, :, i:i + 1, :, :]
            train_cut = train_cut.permute(0, 2, 1, 3, 4)
            train_cut = self.vertical_first(train_cut)
            train_cut = train_cut + self.vertical(train_cut)  # (7,7,32,32)
            vertical[:, :, :, i:i + 1, :, :] = train_cut.view(batch_size, self.n_feats, view_n, 1,
                                                              int(image_h), int(image_w))
        vertical = vertical.view(-1, self.n_feats, view_n * view_n, image_h, image_w)  # (1,49,64,64)

        ''' super-resolution 45'''
        s45 = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)), dtype=torch.float32)
        s45 = s45.to(lr.device)
        position_45 = get_45_position(view_n)
        for item in position_45:
            s45_cut = train_data[:, item[0], item[1], :, :]
            s45_cut = s45_cut.view(batch_size, 1, len(item[0]), image_h, image_w)
            s45_cut = self.s45_first(s45_cut)
            s45_cut = s45_cut + self.s45(s45_cut)
            for i in range(len(item[0])):
                s45[:, :, item[0][i], item[1][i], :, :] = s45_cut[:, :, i, :, :]
        s45 = s45.view(-1, self.n_feats, view_n * view_n, image_h, image_w)

        ''' super-resolution 135'''
        s135 = torch.zeros((batch_size, self.n_feats, view_n, view_n, int(image_h), int(image_w)), dtype=torch.float32)
        s135 = s135.to(lr.device)
        position_135 = get_135_position(view_n)
        for item in position_135:
            s135_cut = train_data[:, item[0], item[1], :, :].view(batch_size, 1, len(item[0]), image_h, image_w)
            s135_cut = self.s135_first(s135_cut)
            s135_cut = s135_cut + self.s135(s135_cut)
            for i in range(len(item[0])):
                s135[:, :, item[0][i], item[1][i], :, :] = s135_cut[:, :, i, :, :]

        s135 = s135.view(-1, self.n_feats, view_n * view_n, image_h, image_w)

        # residual part
        train_data = rearrange(train_data, 'b a1 a2 h w -> b 1 (a1 a2) h w')
        # train_data = train_data.view(batch_size, 1, view_n * view_n, int(image_h), int(image_w))
        train_data = self.upsample_main(train_data)

        full_up = torch.cat((horizontal, vertical, s45, s135), 1)  # (4*n_feats,49,64,64)

        for i in range(self.n_sab):
            full_up = self.s_body_list[i](full_up)
            full_up = full_up.permute(0, 1, 3, 4, 2)
            full_up = full_up.view(-1, 4 * self.n_feats, image_h * image_w, view_n, view_n)  # 4
            full_up = self.a_body_list[i](full_up)
            full_up = full_up.permute(0, 1, 3, 4, 2)
            full_up = full_up.view(-1, 4 * self.n_feats, view_n * view_n, image_h, image_w)  # 4


        full_up = self.upsample(full_up)
        full_up += train_data
        out = rearrange(full_up, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        # full_up = full_up.view(-1, view_n, view_n, image_h * self.scale, image_w * self.scale)  # (7,7,h,w)->(1,49,h,w)

        return out



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class branch_block_2d(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel_size=(1, 3, 3), padding=(0, 1, 1)):
        super(branch_block_2d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        return self.PRelu(self.conv(x))


class FE_block_2d(nn.Module):
    def __init__(self, in_channel=32):
        super(FE_block_2d, self).__init__()
        self.in_channel = in_channel
        assert in_channel % 4 == 0
        self.out_channel = in_channel // 4

        self.uv_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(3, 3, 1),
                                       padding=(1, 1, 0))
        self.xy_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(1, 3, 3),
                                       padding=(0, 1, 1))
        self.ux_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(3, 3, 1),
                                       padding=(1, 1, 0))
        self.vy_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(1, 3, 3),
                                       padding=(0, 1, 1))

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)

        x = x.reshape(batch, channel, height_view, width_view, height * width)
        uv_data = self.uv_conv(x)
        uv_data = uv_data.reshape(batch, channel // 4, height_view, width_view, height, width)

        x = x.reshape(batch, channel, height_view * width_view, height, width)
        xy_data = self.xy_conv(x)
        xy_data = xy_data.reshape(batch, channel // 4, height_view, width_view, height, width)

        x = x.reshape(batch, channel, height_view, width_view, height, width)
        x = x.permute(0, 1, 2, 4, 3, 5)

        x = x.reshape(batch, channel, height_view, height, width_view * width)
        ux_data = self.ux_conv(x)
        ux_data = ux_data.reshape(batch, channel // 4, height_view, height, width_view, width)
        ux_data = ux_data.permute(0, 1, 2, 4, 3, 5)

        x = x.reshape(batch, channel, height_view * height, width_view, width)
        vy_data = self.vy_conv(x)
        vy_data = vy_data.reshape(batch, channel // 4, height_view, height, width_view, width)
        vy_data = vy_data.permute(0, 1, 2, 4, 3, 5)
        del x

        return torch.cat([uv_data, xy_data, ux_data, vy_data], dim=1)


class branch_block_3d(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel_size=3, padding=1):
        super(branch_block_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        return self.PRelu(self.conv(x))


class FE_block_3d(nn.Module):
    def __init__(self, in_channel=32):
        super(FE_block_3d, self).__init__()
        self.in_channel = in_channel
        assert in_channel % 4 == 0
        self.out_channel = in_channel // 4

        self.uvx_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)
        self.uvy_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)
        self.uxy_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)
        self.vxy_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        channel = channel // 4

        uvx_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(width):
            uvx_data[:, :, :, :, :, i] = self.uvx_conv(x[:, :, :, :, :, i])

        uvy_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(height):
            uvy_data[:, :, :, :, i, :] = self.uvy_conv(x[:, :, :, :, i, :])

        uxy_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(width_view):
            uxy_data[:, :, :, i, :, :] = self.uxy_conv(x[:, :, :, i, :, :])

        vxy_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(height_view):
            vxy_data[:, :, i, :, :, :] = self.vxy_conv(x[:, :, i, :, :, :])

        return torch.cat([uvx_data, uvy_data, uxy_data, vxy_data], dim=1)


class res_block_2d(nn.Module):
    def __init__(self, channel):
        super(res_block_2d, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        x = x + self.PRelu(self.conv(x))
        return x


class ResBlock3d(nn.Module):
    def __init__(self, n_feats, kernel_size, padding=(2, 1, 1), bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock3d, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv3d(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class ResBlockc3d(nn.Module):
    def __init__(self, n_feats, bias=True, res_scale=1, is_large_kernel=False):
        super(ResBlockc3d, self).__init__()
        if is_large_kernel:
            kernel_size = (5, 3, 3)
            padding = (2, 1, 1)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        m = []

        m.append(nn.PReLU())
        # m.append(nn.ReLU())
        # m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias))
        # m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias))

        m.append(nn.Conv3d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class ResBlock_SA_2d(nn.Module):
    def __init__(self, n_feats, padding=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_SA_2d, self).__init__()
        m = []
        m.append(nn.PReLU())
        m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=bias))

        n = []
        n.append(nn.PReLU())
        n.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=bias))

        self.s_body = nn.Sequential(*m)
        self.a_body = nn.Sequential(*n)
        self.res_scale = res_scale
        self.view_n = None
        self.image_h = None
        self.image_w = None
        self.n_feats = n_feats

    def forward(self, x):
        res = self.s_body(x)
        # res = res.view(-1, self.n_feats, self.view_n, self.view_n, self.image_h, self.image_w)
        res = res.permute(0, 1, 3, 4, 2)
        res = res.view(-1, self.n_feats, self.image_h * self.image_w, self.view_n, self.view_n)
        res = self.a_body(res)
        # res = res.view(-1, self.n_feats, self.image_h, self.image_w, self.view_n, self.view_n, )
        res = res.permute(0, 1, 3, 4, 2)
        res = res.view(-1, self.n_feats, self.view_n * self.view_n, self.image_h, self.image_w)
        res += x

        return res


class ResBlock2d(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, padding=1, bias=True, bn=False, act=nn.ReLU(True),
                 res_scale=1):
        super(ResBlock2d, self).__init__()
        m = []
        m.append(nn.Conv3d(in_feats, out_feats, kernel_size, padding=[0, 1, 1], bias=bias))
        m.append(nn.PReLU())
        # m.append(nn.ReLU())

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)

        return res


class Block2d(nn.Module):
    def __init__(self, n_feats, kernel_size, padding=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(Block2d, self).__init__()
        m = []
        m.append(nn.Conv3d(n_feats, n_feats // 2, kernel_size, padding=[0, 1, 1], bias=bias))
        m.append(nn.PReLU())

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def get_45_position(view_n):
    start_position_list = []
    for i in range(view_n):
        start_position_list.append(([i], [0]))
    for j in range(1, view_n):
        start_position_list.append(([view_n - 1], [j]))
    for item in start_position_list:
        while item[0][0] > 0 and item[1][0] < view_n - 1:
            item[0].insert(0, item[0][0] - 1)
            item[1].insert(0, item[1][0] + 1)
    return start_position_list

def get_135_position(view_n):
    start_position_list = []
    for i in range(view_n):
        start_position_list.append(([i], [0]))
    for j in range(1, view_n):
        start_position_list.append(([0], [j]))
    for item in start_position_list:
        while item[0][-1] < view_n - 1 and item[1][-1] < view_n - 1:
            item[0].append(item[0][-1] + 1)
            item[1].append(item[1][-1] + 1)
    return start_position_list














class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, info=None):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):

    pass
