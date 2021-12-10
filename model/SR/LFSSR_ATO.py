'''
@inproceedings{LFSSR_ATO,
  title={Light field spatial super-resolution via deep combinatorial geometry embedding and structural consistency regularization},
  author={Jin, Jing and Hou, Junhui and Chen, Jie and Kwong, Sam},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2260--2269},
  year={2020}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as functional
import math


class get_model(nn.Module):
    def __init__(self, opt):
        super(get_model, self).__init__()
        
        channels = 64
        self.angRes = 5
        self.an2 = self.angRes ** 2
        self.scale = opt.scale_factor
        
        self.fea_conv0 = nn.Conv2d(1, channels, 3, 1, 1, bias=True)
        self.fea_resblock = make_layer(ResidualBlock, channels, n_layers=5)
        
        self.pair_conv0 = nn.Conv2d(2*channels, channels, 3, 1, 1, bias=True)
        self.pair_resblock = make_layer(ResidualBlock, channels, n_layers=2)
        self.pair_conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        self.fusion_view_conv0 = nn.Conv2d(self.an2, channels, 3, 1, 1, bias=True)
        self.fusion_view_resblock = make_layer(ResidualBlock, channels, n_layers=2)
        self.fusion_view_conv1 = nn.Conv2d(channels, 1, 3, 1, 1, bias=True)
        
        self.fusion_fea_conv0 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.fusion_fea_resblock = make_layer(ResidualBlock, channels, n_layers=3)
       
        up = []
        for _ in range(int(math.log(self.scale,2))):
            up.append(nn.Conv2d(channels, 4*channels, 3, 1, 1, bias=True))
            up.append(nn.PixelShuffle(2))
            up.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*up)
        
        self.HRconv = nn.Conv2d(channels, channels//2, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(channels//2, 1, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, lf_lr, angRes):
        ######################################################
        # reshape 成原作者网络的方法，将angular dimension 单独成一维
        [B, _, H, W] = lf_lr.size()
        lf_lr = lf_lr.view(B, 1, self.angRes, H//self.angRes, self.angRes, W//self.angRes).permute(0,1,2,4,3,5)
        lf_lr = lf_lr.contiguous().view(B, self.an2, H//self.angRes, W//self.angRes) # [B, angRes**2, h, w]
        B, an2, h, w = lf_lr.size()
        ######################################################

        # bilinear
        lf_lr_upsample = functional.interpolate(lf_lr, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # individual LR feature extraction 对子图像做特征提取，子图像之间不做交互
        lf_fea = self.relu(self.fea_conv0(lf_lr.view(-1, 1, h, w)))
        lf_fea = self.fea_resblock(lf_fea).view(B, an2, -1, h, w)  # [B,an2,channels,h,w]

        # 预置输出
        out = []
        for i in range(self.an2):
            # 构造 pair feature
            lf_pair_fea = torch.cat([lf_fea, lf_fea[:, i:i+1, :, :, :].repeat(1, an2, 1, 1, 1)], dim=2)  # [N,an2,128,h, w]

            # pair fusion 将所有视图与参考视图的特征融合
            lf_pair_fea = self.relu(self.pair_conv0(lf_pair_fea.view(B * an2, -1, h, w)))
            lf_pair_fea = self.pair_resblock(lf_pair_fea)
            lf_fea_aligned = self.pair_conv1(lf_pair_fea)  # [N*an2,64,h, w]
            lf_fea_aligned = lf_fea_aligned.view(B, an2, -1, h, w)  # [N, an2, channels, h, w]

            # all view fusion 将所有视图做特征融合 到一个视图
            lf_fea_aligned = torch.transpose(lf_fea_aligned, 1, 2).contiguous()  # [B, channels, an2,h,w]
            ref_fea_fused = self.relu(self.fusion_view_conv0(lf_fea_aligned.view(-1, an2, h, w)))  # [N*channels,64,h,w]
            ref_fea_fused = self.fusion_view_resblock(ref_fea_fused)  # [B*channels,64,h,w]
            ref_fea_fused = self.relu(self.fusion_view_conv1(ref_fea_fused))  # [B*channels,1,h,w]
            ref_fea_fused = ref_fea_fused.view(B, -1, h, w)  # [B,channels,h,w]
            ref_fea_fused = self.relu(self.fusion_fea_conv0(ref_fea_fused))  # [B,channels,h,w]
            ref_fea_fused = self.fusion_fea_resblock(ref_fea_fused)  # [B, channels,h,w]

            # upsample
            ref_fea_hr = self.upsampler(ref_fea_fused) # [B, channels,h*scale,w*scale]
            out_one_view = self.relu(self.HRconv(ref_fea_hr))
            out_one_view = self.conv_last(out_one_view)  # [B,1,h*scale_factor,w*scale_factor]
            out.append(out_one_view)
            pass

        out = torch.cat(out, 1)  # [N,an2,h,w]

        # 增加全局残差
        out += lf_lr_upsample

        # reshape for output
        out = out.view(B, self.angRes, self.angRes, h*self.scale, w*self.scale).permute(0,1,3,2,4)
        out = out.contiguous().view(B, 1, self.angRes*h*self.scale, self.angRes*w*self.scale)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)


class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()

        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.relu(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.view(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * h * w, c, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.relu(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.view(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * self.an * self.an, c, h, w)  # [N*an2,c,h,w]

        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss

def weights_init(m):
    pass
