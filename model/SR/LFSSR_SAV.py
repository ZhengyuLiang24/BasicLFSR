'''
@article{cheng2022spatial,
  title={Spatial-Angular Versatile Convolution for Light Field Reconstruction},
  author={Cheng, Zhen and Liu, Yutong and Xiong, Zhiwei},
  journal={IEEE Transactions on Computational Imaging},
  volume={8},
  pages={1131--1144},
  year={2022},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable
import torch.nn.functional as F
import math
from einops import rearrange


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.scale = args.scale_factor
        self.angRes = args.angRes_in

        if self.scale == 2:
            self.net = net2x(an=args.angRes_in, layer=16, mode='parares', fn=45)
        elif self.scale == 4:
            self.net = net4x(an=args.angRes_in, layer=10, mode='parares', fn=45)

    def forward(self, x, info=None):
        # rgb2ycbcr
        x = rearrange(x, 'b 1 (u h) (v w) -> b (u v) h w', u=self.angRes, v=self.angRes)

        y = self.net(x)
        y = rearrange(y, 'b (u v) h w -> b 1 (u h) (v w)', u=self.angRes, v=self.angRes)

        return y


class net2x(nn.Module):
    def __init__(self, an, layer, mode="catres", fn=64):
        super(net2x, self).__init__()
        self.an = an
        self.an2 = an * an
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=fn, kernel_size=3, stride=1, padding=1)

        #
        if mode == "catres":
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para) for _ in range(layer)]
        elif mode == 'cat':
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=False) for _ in range(layer)]
        elif mode == 'Dserial':
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_double_serial(SAS_para=sas_para, SAC_para=sac_para) for _ in range(layer)]
        elif mode == "parares":
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False) for _ in
                          range(layer)]
        elif mode == "SAS":
            alt_blocks = [SAS_conv(act='lrelu', fn=fn) for _ in range(layer)]
        elif mode == "SAC":
            alt_blocks = [SAC_conv(act='lrelu', fn=fn) for _ in range(layer)]
        else:
            raise Exception("Wrong mode!")
        self.refine_sas = nn.Sequential(*alt_blocks)

        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=fn, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_in', 'leaky_relu')
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, lr):
        N, dimsize, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]
        x = self.lrelu(self.conv0(lr))  # [N*81,64,h,w]

        ## CS-MOD
        lf_out = x.view(N, self.an2, -1, h, w)  # [N, an2, C, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
        lf_out = lf_out.view(N, lf_out.shape[1], self.an, self.an, h, w)  # [N, C, an, an, H, W]
        lf_out = self.refine_sas(lf_out)  # [N*81,64,h,w]
        lf_out = lf_out.view(N, -1, self.an2, h, w)  # [N, C, an2, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
        lf_out = lf_out.view(N * self.an2, lf_out.shape[2], h, w)

        fup_1 = self.fup1(lf_out)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]
        sr_2x = sr_2x.view(N, self.an2, h * 2, w * 2)
        return sr_2x


class net4x(nn.Module):
    def __init__(self, an, layer, mode="catres", fn=64):
        super(net4x, self).__init__()
        self.an = an
        self.an2 = an * an
        self.fn = fn
        self.mode = mode
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=fn, kernel_size=3, stride=1, padding=1)

        self.altblock1 = self.make_layer(layer_num=layer, fn=fn)

        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=fn, out_channels=fn, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=fn, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.altblock2 = self.make_layer(layer_num=layer, fn=fn)

        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=fn, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.res2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_in', 'leaky_relu')
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, layer_num=6, fn=64):
        layers = []
        sas_para = SAS_para()
        sac_para = SAC_para()
        sas_para.fn = fn
        sac_para.fn = fn
        if self.mode == "catres":
            for i in range(layer_num):
                layers.append(SAV_concat(SAS_para=sas_para, SAC_para=sac_para))
        elif self.mode == "parares":
            for i in range(layer_num):
                layers.append(SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False))
        elif self.mode == "Dserial":
            for i in range(layer_num):
                layers.append(SAV_double_serial(SAS_para=sas_para, SAC_para=sac_para))
        elif self.mode == "SAS":
            for i in range(layer_num):
                layers.append(SAS_conv(act='lrelu', fn=fn))
        elif self.mode == "SAC":
            for i in range(layer_num):
                layers.append(SAC_conv(act='lrelu', fn=fn))
        return nn.Sequential(*layers)

    def forward(self, lr):
        N, _, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.lrelu(self.conv0(lr))  # [N*81,64,h,w]
        #########
        lf_out = x.view(N, self.an2, -1, h, w)  # [N, an2, C, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
        lf_out = lf_out.view(N, lf_out.shape[1], self.an, self.an, h, w)  # [N, C, an, an, H, W]
        lf_out = self.altblock1(lf_out)  # [N*81,64,h,w]
        lf_out = lf_out.view(N, -1, self.an2, h, w)  # [N, C, an2, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
        lf_out = lf_out.view(N * self.an2, lf_out.shape[2], h, w)
        #######
        fup_1 = self.fup1(lf_out)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]
        ##########
        f_2 = fup_1.view(N, self.an2, -1, 2 * h, 2 * w)  # [N, an2, C, H, W]
        f_2 = f_2.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
        f_2 = f_2.view(N, f_2.shape[1], self.an, self.an, 2 * h, 2 * w)  # [N, C, an, an, H, W]
        f_2 = self.altblock2(f_2)  # [N*81,64,2h,2w]
        f_2 = f_2.view(N, -1, self.an2, 2 * h, 2 * w)  # [N, C, an2, H, W]
        f_2 = f_2.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
        f_2 = f_2.view(N * self.an2, f_2.shape[2], 2 * h, 2 * w)
        ##########
        fup_2 = self.fup2(f_2)  # [N*81,64,4h,4w]
        res_2 = self.res2(fup_2)  # [N*81,1,4h,4w]
        iup_2 = self.iup2(sr_2x)  # [N*81,1,4h,4w]
        sr_4x = res_2 + iup_2  # [N*81,1,4h,4w]

        sr_2x = sr_2x.view(N, self.an2, h * 2, w * 2)
        sr_4x = sr_4x.view(N, self.an2, h * 4, w * 4)

        return sr_4x


class SAS_para:
    def __init__(self):
        self.act = 'lrelu'
        self.fn = 64


class SAC_para:
    def __init__(self):
        self.act = 'lrelu'
        self.symmetry = True
        self.max_k_size = 3
        self.fn = 64


class SAS_conv(nn.Module):
    def __init__(self, act='relu', fn=64):
        super(SAS_conv, self).__init__()
        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        self.spaconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.spaconv.weight, a, 'fan_in', self.init_indicator)
        nn.init.constant_(self.spaconv.bias, 0.0)

        self.angconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.angconv.weight, a, 'fan_in', self.init_indicator)
        nn.init.constant_(self.angconv.bias, 0.0)

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(N * U * V, c, h, w)

        out = self.act(self.spaconv(x))  # [N*U*V,c,h,w]
        out = out.view(N, U * V, c, h * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * h * w, c, U, V)  # [N*h*w,c,U,V]

        out = self.act(self.angconv(out))  # [N*h*w,c,U,V]
        out = out.view(N, h * w, c, U * V)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, U, V, c, h, w)  # [N,U,V,c,h,w]
        out = out.permute(0, 3, 1, 2, 4, 5).contiguous()  # [N,c,U,V,h,w]

        return out


class SAC_conv(nn.Module):
    def __init__(self, act='relu', symmetry=True, max_k_size=3, fn=64):
        super(SAC_conv, self).__init__()
        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        if symmetry:
            k_size_ang = max_k_size
            k_size_spa = max_k_size
        else:
            k_size_ang = max_k_size - 2
            k_size_spa = max_k_size

        self.verconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1, 1), padding=(k_size_ang // 2, k_size_spa // 2))
        nn.init.kaiming_normal_(self.verconv.weight, a, 'fan_in', self.init_indicator)
        nn.init.constant_(self.verconv.bias, 0.0)

        self.horconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1, 1), padding=(k_size_ang // 2, k_size_spa // 2))
        nn.init.kaiming_normal_(self.horconv.weight, a, 'fan_in', self.init_indicator)
        nn.init.constant_(self.horconv.bias, 0.0)

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N * V * w, c, U, h)

        out = self.act(self.verconv(x))  # [N*V*w,c,U,h]
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * U * h, c, V, w)  # [N*U*h,c,V,w]

        out = self.act(self.horconv(out))  # [N*U*h,c,V,w]
        out = out.view(N, U * h, c, V * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, V, w, c, U, h)  # [N,V,w,c,U,h]
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()  # [N,c,U,V,h,w]
        return out


class SAV_concat(nn.Module):
    def __init__(self, SAS_para, SAC_para, residual_connection=True):
        """
        parameters for building SAS-SAC block
        :param SAS_para: {act, fn}
        :param SAC_para: {act, symmetry, max_k_size, fn}
        :param residual_connection: True or False for residual connection
        """
        super(SAV_concat, self).__init__()
        self.res_connect = residual_connection
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)

    def forward(self, lf_input):
        feat = self.SAS_conv(lf_input)
        res = self.SAC_conv(feat)
        if self.res_connect:
            res += lf_input
        return res


class SAV_double_serial(nn.Module):
    def __init__(self, SAS_para, SAC_para):
        super(SAV_double_serial, self).__init__()
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size,
                                 fn=SAC_para.fn)

    def forward(self, lf_input):
        res = self.SAS_conv(lf_input) + lf_input
        res = self.SAC_conv(res) + res
        return res


class SAV_parallel(nn.Module):
    def __init__(self, SAS_para, SAC_para, feature_concat=True):
        super(SAV_parallel, self).__init__()
        self.feature_concat = feature_concat
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)
        self.lrelu_para = nn.LeakyReLU(negative_slope=0.2, inplace=True)  #
        if self.feature_concat:
            self.channel_reduce = convNd(in_channels=2 * SAS_para.fn,
                                         out_channels=SAS_para.fn,
                                         num_dims=4,
                                         kernel_size=(1, 1, 1, 1),
                                         stride=(1, 1, 1, 1),
                                         padding=(0, 0, 0, 0),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in', 'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))

    def forward(self, lf_input):
        sas_feat = self.SAS_conv(lf_input)
        # sas_feat = self.lrelu_para(sas_feat)#
        sac_feat = self.SAC_conv(lf_input)  # [N,c,U,V,h,w]
        # sac_feat = self.lrelu_para(sac_feat)

        if self.feature_concat:
            concat_feat = torch.cat((sas_feat, sac_feat), dim=1)  # [N,2c,U,V,h,w]
            feat = self.lrelu_para(concat_feat)
            res = self.channel_reduce(feat)
            res += lf_input
        else:
            res = sas_feat + sac_feat + lf_input
        return res


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


class convNd(nn.Module):
    """Some Information about convNd"""
    def __init__(self, in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size: Tuple,
                 stride,
                 padding,
                 is_transposed=False,
                 padding_mode='zeros',
                 output_padding=0,
                 dilation: int = 1,
                 groups: int = 1,
                 rank: int = 0,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(convNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, Tuple):
            padding = tuple(padding for _ in range(num_dims))
        if not isinstance(output_padding, Tuple):
            output_padding = tuple(output_padding for _ in range(num_dims))
        if not isinstance(dilation, Tuple):
            dilation = tuple(dilation for _ in range(num_dims))

        # This parameter defines which Pytorch convolution to use as a base, for 3 Conv2D is used
        if rank == 0 and num_dims <= 3:
            max_dims = num_dims - 1
        else:
            max_dims = 3

        if is_transposed:
            self.conv_f = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[max_dims - 1]
        else:
            self.conv_f = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[max_dims - 1]

        assert len(kernel_size) == num_dims, \
            'nD kernel size expected!'
        assert len(stride) == num_dims, \
            'nD stride size expected!'
        assert len(padding) == num_dims, \
            'nD padding size expected!'
        assert len(output_padding) == num_dims, \
            'nD output_padding size expected!'
        assert sum(dilation) == num_dims, \
            'Dilation rate other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.rank = rank
        self.is_transposed = is_transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.bias_initializer is not None:
            if self.use_bias:
                self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0]

        for _ in range(next_dim_len):
            if self.num_dims - 1 > max_dims:
                # Initialize a Conv_n-1_D layer
                conv_layer = convNd(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    use_bias=self.use_bias,
                                    num_dims=self.num_dims - 1,
                                    rank=self.rank - 1,
                                    is_transposed=self.is_transposed,
                                    kernel_size=self.kernel_size[1:],
                                    stride=self.stride[1:],
                                    groups=self.groups,
                                    dilation=self.dilation[1:],
                                    padding=self.padding[1:],
                                    padding_mode=self.padding_mode,
                                    output_padding=self.output_padding[1:],
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

            else:
                # Initialize a Conv layer
                # bias should only be applied by the top most layer, so we disable bias in the internal convs
                conv_layer = self.conv_f(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         bias=False,
                                         kernel_size=self.kernel_size[1:],
                                         dilation=self.dilation[1:],
                                         stride=self.stride[1:],
                                         padding=self.padding[1:],
                                         # padding_mode=self.padding_mode,
                                         groups=self.groups)
                if self.is_transposed:
                    conv_layer.output_padding = self.output_padding[1:]

                # Apply initializer functions to weight and bias tensor
                if self.kernel_initializer is not None:
                    self.kernel_initializer(conv_layer.weight)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------
    def forward(self, input):

        # Pad the input if is not transposed convolution
        if not self.is_transposed:
            padding = list(self.padding)
            # Pad input if this is the parent convolution ie rank=0
            if self.rank == 0:
                inputShape = list(input.shape)
                inputShape[2] += 2 * self.padding[0]
                padSize = (0, 0, self.padding[0], self.padding[0])
                padding[0] = 0
                if self.padding_mode == 'zeros':
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize, 'constant',
                                  0).view(inputShape)
                else:
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize,
                                  self.padding_mode).view(inputShape)

        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        if not self.is_transposed:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [math.floor((size_i[x] + 2 * padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
            # Compute size of the output without stride
            size_ons = tuple([size_i[x] - size_k[x] + 1 for x in range(len(size_i))])
        else:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [(size_i[x] - 1) * self.stride[x] - 2 * self.padding[x] + (size_k[x] - 1) + 1 + self.output_padding[x]
                 for x in range(len(size_i))])

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b, self.out_channels) + size_o[1:], device=input.device)]
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate inputs first dimmension
            for j in range(size_i[0]):

                # Add results to this output frame
                if self.is_transposed:
                    out_frame = i + j * self.stride[0] - self.padding[0]
                else:
                    out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_ons[0]) // 2 - (1 - size_k[0] % 2)
                    k_center_position = out_frame % self.stride[0]
                    out_frame = math.floor(out_frame / self.stride[0])
                    if k_center_position != 0:
                        continue

                if out_frame < 0 or out_frame >= size_o[0]:
                    continue

                # Prepate input for next dimmension
                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = \
                    self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b, resultShape[1], -1)
            for k in range(self.out_channels):
                result[:, k, :] += self.bias[k]
            return result.view(resultShape)
        else:
            return result


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass

