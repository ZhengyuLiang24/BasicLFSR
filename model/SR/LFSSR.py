'''
@article{LFSSR,
  title={Light field spatial super-resolution using deep efficient spatial-angular separable convolution},
  author={Yeung, Henry Wing Fung and Hou, Junhui and Chen, Xiaoming and Chen, Jie and Chen, Zhibo and Chung, Yuk Ying},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={5},
  pages={2319--2330},
  year={2018},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        self.factor = args.scale_factor
        if args.scale_factor==2:
            self.net = net2x(self.angRes)
        elif args.scale_factor==4:
            self.net = net4x(self.angRes)


    def forward(self, lr, Lr_Info):
        # reshape for LFSSR
        B, _, H, W = lr.shape
        h = H // self.angRes
        w = W // self.angRes
        lr = lr.view(B, 1, self.angRes, h, self.angRes, w)
        lr = lr.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, angRes, angRes, 1, h, w)
        lr = lr.view(B, self.angRes * self.angRes, h, w)  # [B*angRes**2, 1, h, w]

        # LFSSR reconstruction
        out = self.net(lr)

        # reshape for output
        out = out.view(B, self.angRes, self.angRes, h*self.factor, w*self.factor)
        out = out.permute(0,1,3,2,4).contiguous()
        out = out.view(B, 1, self.angRes*h*self.factor, self.angRes*w*self.factor)
        return out



class net2x(nn.Module):
    def __init__(self, an):
        super(net2x, self).__init__()

        self.an = an
        self.an2 = an * an
        layer = 10
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.altblock1 = self.make_layer(layer_num=layer)
        self.fup1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * 2 ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1 * 2 ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.contiguous().view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(AltFilter(self.an))
        return nn.Sequential(*layers)

    def forward(self, lr):

        N, _, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.contiguous().view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.relu(self.conv0(lr))  # [N*81,64,h,w]
        f_1 = self.altblock1(x)  # [N*81,64,h,w]
        fup_1 = self.fup1(f_1)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]
        sr_2x = sr_2x.contiguous().view(N, self.an2, h * 2, w * 2)
        return sr_2x


class net4x(nn.Module):

    def __init__(self, an):

        super(net4x, self).__init__()

        self.an = an
        self.an2 = an * an
        layer = 10
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.altblock1 = self.make_layer(layer_num=layer)

        self.fup1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * 2 ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1 * 2 ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

        self.altblock2 = self.make_layer(layer_num=layer)
        self.fup2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * 2 ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1 * 2 ** 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.contiguous().view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(AltFilter(self.an))
        return nn.Sequential(*layers)

    def forward(self, lr):

        N, _, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.contiguous().view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.relu(self.conv0(lr))  # [N*81,64,h,w]
        f_1 = self.altblock1(x)  # [N*81,64,h,w]
        fup_1 = self.fup1(f_1)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]

        f_2 = self.altblock2(fup_1)  # [N*81,64,2h,2w]
        fup_2 = self.fup2(f_2)  # [N*81,64,4h,4w]
        res_2 = self.res2(fup_2)  # [N*81,1,4h,4w]
        iup_2 = self.iup2(sr_2x)  # [N*81,1,4h,4w]
        sr_4x = res_2 + iup_2  # [N*81,1,4h,4w]

        sr_2x = sr_2x.contiguous().view(N, self.an2, h * 2, w * 2)
        sr_4x = sr_4x.contiguous().view(N, self.an2, h * 4, w * 4)

        return sr_4x


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)

    return torch.from_numpy(filter).float()


class AltFilter(nn.Module):
    def __init__(self, angRes):
        super(AltFilter, self).__init__()

        self.angRes = angRes
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.angRes * self.angRes)

        out = self.relu(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.contiguous().view(N, self.angRes * self.angRes, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.contiguous().view(N * h * w, c, self.angRes, self.angRes)  # [N*h*w,c,an,an]

        out = self.relu(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.contiguous().view(N, h * w, c, self.angRes * self.angRes)
        out = torch.transpose(out, 1, 3)
        out = out.contiguous().view(N * self.angRes * self.angRes, c, h, w)  # [N*an2,c,h,w]
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

