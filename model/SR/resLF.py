'''
@inproceedings{resLF,
  title={Residual networks for light field image super-resolution},
  author={Zhang, Shuo and Lin, Youfang and Sheng, Hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11046--11055},
  year={2019}
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
        self.factor = args.scale_factor
        self.angRes = args.angRes_in

        self.net_side = basic_Net(3, self.factor)
        self.net_corner = basic_Net(3, self.factor)
        self.net_3x3 = basic_Net(3, self.factor)
        self.net_5x5 = basic_Net(5, self.factor)
        self.net_7x7 = basic_Net(7, self.factor)
        self.net_9x9 = basic_Net(9, self.factor)





    def forward(self, x, LR_info):
        angRes = self.angRes
        [B, _, H, W] = x.size()
        h = H//angRes
        w = W//angRes

        ''' 预设输出 '''
        x = x.view(B, 1, angRes, h, angRes, w).permute(0,1,2,4,3,5) # [B, 1, angRes, angRes, h, w]
        x_padding = torch.zeros((B, 1, angRes+2, angRes+2, h, w)).cuda() # [B, 1, angRes+2, angRes+2, h, w]
        x_padding[:, :, 1:angRes+1, 1:angRes+1, :, :] = x[:,:,:,:,:,:]
        out = torch.zeros(B, 1, angRes, angRes, h*self.factor, w*self.factor).cuda()

        ''' 根据 angRes 设置不同的子网络'''
        if angRes==3:
            sub_net = [self.net_3x3]
        elif angRes==5:
            sub_net = [self.net_5x5, self.net_3x3]
        elif angRes==7:
            sub_net = [self.net_7x7, self.net_5x5, self.net_3x3]
        elif angRes==9:
            sub_net = [self.net_9x9, self.net_7x7, self.net_5x5, self.net_3x3]
        else:
            sub_net = []

        for i in range(angRes):
            for j in range(angRes):
                distance = sqrt((i-angRes//2)**2 + (j-angRes//2)**2)
                for threshold in range(angRes//2):
                    if distance >= (angRes//2):
                        # 判断为 corner 或 side
                        tmp_x = x_padding[:, :, i :i + 3, j :j + 3, :, :]
                        if (i,j)==(0,0) or (i,j)==(0, angRes-1) or (i,j)==(angRes-1, 0) or (i,j)==(angRes-1, angRes-1):
                            ''' corner '''
                            out[:, :, i, j, :, :] = self.net_corner(tmp_x, LR_info)
                            break
                        else:
                            ''' side '''
                            out[:, :, i, j, :, :] = self.net_side(tmp_x, LR_info)
                            break

                    elif distance <= 0*sqrt(2):
                        # 中心视角
                        # print('central')
                        tmp_net = sub_net[0]
                        tmp_x = x[:,:, :, :, :, :]
                        out[:, :, i, j, :, :] = tmp_net(tmp_x, LR_info)
                        break
                    elif distance <= 1*sqrt(2):
                        # print('central-1')
                        tmp_net = sub_net[1]
                        tmp_radius = angRes//2 - 1
                        tmp_x = x[:,:, i-tmp_radius:i+tmp_radius+1, j-tmp_radius:j+tmp_radius+1, :, :]
                        out[:, :, i, j, :, :] = tmp_net(tmp_x, LR_info)
                        break
                    elif distance <= 2*sqrt(2):
                        # print('central-2')
                        tmp_net = sub_net[2]
                        tmp_radius = angRes // 2 - 2
                        tmp_x = x[:, :, i - tmp_radius:i + tmp_radius + 1, j - tmp_radius:j + tmp_radius + 1, :, :]
                        out[:, :, i, j, :, :] = tmp_net(tmp_x, LR_info)
                        break
                    elif distance <= 3*sqrt(2):
                        # print('central-3')
                        tmp_net = sub_net[3]
                        tmp_radius = angRes // 2 - 3
                        tmp_x = x[:, :, i - tmp_radius:i + tmp_radius + 1, j - tmp_radius:j + tmp_radius + 1, :, :]
                        out[:, :, i, j, :, :] = tmp_net(tmp_x, LR_info)
                        break
                pass
            pass

        ''' 中心视角输出 '''
        out = out.permute(0,1,2,4,3,5).contiguous().view(B,1, H*self.factor, W*self.factor)

        return out


class basic_Net(nn.Module):
    def __init__(self, radius, scale_factor):
        super(basic_Net, self).__init__()
        self.radius = radius
        self.scale_factor = scale_factor
        channels = 32

        self.central_head = nn.Conv2d(1, channels, kernel_size=3, padding=1, stride=1, bias=False)

        self.head = nn.Conv2d(self.radius, channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.midbody = self.make_layer(ResBlock, 4, channels)

        self.body = self.make_layer(ResBlock, 4, channels*4)
        self.body_degrade = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, stride=1,bias=False)

        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels*self.scale_factor**2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )


    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x, LR_info):
        if len(x.size()) == 4:
            angRes = self.angRes
            [B, _, H, W] = x.size()
            h = H // angRes
            w = W // angRes
            x = x.view(B, 1, angRes, h, angRes, w).permute(0, 1, 2, 4, 3, 5)

        [B, _, _, _, h, w] = x.size()
        radius = self.radius
        # 提前中心视角子图像
        central_x = x[:, :, radius // 2, radius // 2, :, :]

        # 提取 0 度方向的子视图
        train_data_0 = torch.zeros(B, radius, h, w).cuda()
        i = int(radius // 2)
        for j in range(radius):
            train_data_0[:, j:j + 1, :, :] = x[:, :, i, j, :, :]
            pass

        # 提取 90 度方向的子视图
        train_data_90 = torch.zeros(B, radius, h, w).cuda()
        j = int(radius // 2)
        for i in range(radius):
            train_data_90[:, i:i + 1, :, :] = x[:, :, i, j, :, :]

        # 提取 45 度方向的子视图
        train_data_45 = torch.zeros(B, radius, h, w).cuda()
        for i in range(radius):
            j = int(radius - i) - 1
            train_data_45[:, i:i + 1, :, :] = x[:, :, i, j, :, :]

        # 提取 135 度方向的子视图
        train_data_135 = torch.zeros(B, radius, h, w).cuda()
        for i in range(radius):
            j = i
            train_data_135[:, i:i + 1, :, :] = x[:, :, i, j, :, :]

        # 特征提取
        # x_upscale = torch.nn.functional.interpolate(central_x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        res_x = self.central_head(central_x)
        mid_0d = self.midbody(self.head(train_data_0))
        mid_90d = self.midbody(self.head(train_data_90))
        mid_45d = self.midbody(self.head(train_data_45))
        mid_135d = self.midbody(self.head(train_data_135))

        ''' Merge layers '''
        mid_merged = torch.cat((mid_0d, mid_90d, mid_45d, mid_135d), 1)
        res = self.body_degrade(self.body(mid_merged))

        res += res_x

        out = self.tail(res) # + x_upscale

        return out

class ResBlock(nn.Module):
    def __init__(self, channels, res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(channels, channels, kernel_size = 3, padding=1, stride=1))
            if i == 0: m.append(nn.ReLU(True))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResBlock, self).__init__()
#         self.body =
#         self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         return self.relu(self.conv(x))


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss

