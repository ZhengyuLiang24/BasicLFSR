'''
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI},
    year      = {2022},
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.angRes = args.angRes_in
        self.factor = args.scale_factor
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=self.angRes, padding=self.angRes, bias=False)
        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(self.factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, info=None):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = SAI2MacPI(x, self.angRes)
        buffer = self.init_conv(x)
        buffer = self.disentg(buffer)
        buffer_SAI = MacPI2SAI(buffer, self.angRes)
        out = self.upsample(buffer_SAI) + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels//4, channels//2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        return buffer + x


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss


def weights_init(m):
    pass


