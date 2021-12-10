'''
@article{IINet,
  title={Intra-Inter View Interaction Network for Light Field Image Super-Resolution},
  author={Liu, Gaosheng and Yue, Huanjing and Wu, Jiamin and Yang, Jingyu},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        n_blocks, channel = 4, 32
        self.factor = args.scale_factor
        self.angRes = args.angRes_in
        self.IntraFeaExtract = FeaExtract(channel)
        self.InterFeaExtract = Extract_inter_fea(channel, self.angRes)
        self.MCB_1 = MCB(channel, self.angRes)
        self.MCB_2 = MCB(channel, self.angRes)
        self.MCB_3 = MCB(channel, self.angRes)
        self.MCB_4 = MCB(channel, self.angRes)
        self.Interact_1 = Intra_inter_FUM(channel, self.angRes)
        self.Interact_2 = Intra_inter_FUM(channel, self.angRes)
        self.Interact_3 = Intra_inter_FUM(channel, self.angRes)
        self.Interact_4 = Intra_inter_FUM(channel, self.angRes, last=True)
        
        self.FBM = FBM(channel*4)
        self.UpSample = Upsample(channel, self.factor)

    def forward(self, x, info=None):
        
        x_multi = LFsplit(x, self.angRes)
        
        intra_fea_initial = self.IntraFeaExtract(x_multi)       
        inter_fea_initial = self.InterFeaExtract(x_multi)
        
        b, n, c, h, w = x_multi.shape
        x_multi = x_multi.contiguous().view(b*n, -1, h, w)
        x_upscale = F.interpolate(x_multi, scale_factor=self.factor, mode='bicubic', align_corners=False)
        _, c, h, w = x_upscale.shape
        x_upscale = x_upscale.unsqueeze(1).contiguous().view(b, -1, c, h, w)
        
        intra_fea_0, inter_fea_1 = self.Interact_1(intra_fea_initial, inter_fea_initial) 
        intra_fea_0 = self.MCB_1(intra_fea_0)
        
        intra_fea_1, inter_fea_2 = self.Interact_2(intra_fea_0.permute(0,2,1,3,4), inter_fea_1)
        intra_fea_1 = self.MCB_2(intra_fea_1)
        
        intra_fea_2, inter_fea_3 = self.Interact_3(intra_fea_1.permute(0,2,1,3,4), inter_fea_2)
        intra_fea_2 = self.MCB_3(intra_fea_2)
        
        intra_fea_3, _ = self.Interact_4(intra_fea_2.permute(0,2,1,3,4), inter_fea_3)
        intra_fea_3 = self.MCB_4(intra_fea_3)
        
        intra_fea = torch.cat((intra_fea_0, intra_fea_1, intra_fea_2, intra_fea_3), 1).permute(0,2,1,3,4)

        intra_fea = self.FBM(intra_fea)
        out_sv = self.UpSample(intra_fea)
        
        out = FormOutput(out_sv) + FormOutput(x_upscale)

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

    def forward(self, x_mv):
        b, n, r, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b*n, -1, h, w)
        intra_fea_0 = self.FEconv(x_mv)
        intra_fea = self.FERB_1(intra_fea_0)
        intra_fea = self.FERB_2(intra_fea)
        intra_fea = self.FERB_3(intra_fea)
        intra_fea = self.FERB_4(intra_fea)
        _, c, h, w = intra_fea.shape
        intra_fea = intra_fea.unsqueeze(1).contiguous().view(b, -1, c, h, w)#.permute(0,2,1,3,4)  # intra_fea:  B, N, C, H, W

        return intra_fea


class Extract_inter_fea(nn.Module):
    def __init__(self, channel, angRes):
        super(Extract_inter_fea, self).__init__()
        self.FEconv = nn.Conv2d(angRes*angRes, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x_mv):
        b, n, r, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b,-1, h, w)
        inter_fea_0 = self.FEconv(x_mv)
        inter_fea = self.FERB_1(inter_fea_0)
        inter_fea = self.FERB_2(inter_fea)
        inter_fea = self.FERB_3(inter_fea)
        inter_fea = self.FERB_4(inter_fea)
        return inter_fea


class Intra_inter_FUM(nn.Module):
    '''
    Inter-assist-intra feature updating module & intra-assist-inter feature updating module 
    '''
    def __init__(self, channel, angRes, last=False):
        super(Intra_inter_FUM, self).__init__()
        self.conv_fusing = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.conv_sharing = nn.Conv2d(angRes*angRes*channel, angRes*angRes*channel, kernel_size=1, stride=1, padding=0)
        self.last = last
        
        if not last:
            self.conv_f1 = nn.Conv2d(angRes*angRes*channel, channel, kernel_size=1, stride=1, padding=0)
            self.conv_f2 = nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, intra_fea, inter_fea):
        #intra_fea = intra_fea.permute(0,2,1,3,4)
        b, n, c, h, w = intra_fea.shape
                
        ##update inter-view feature
        upda_intra_feas = []
        for i in range(n):
            current_sv = intra_fea[:, i, :, :, :].contiguous()
            buffer = torch.cat((current_sv, inter_fea), dim=1)
            
            buffer = self.lrelu(self.conv_fusing(buffer))
            upda_intra_feas.append(buffer)
        upda_intra_feas = torch.cat(upda_intra_feas, dim=1)
        fuse_fea = self.conv_sharing(upda_intra_feas)
        
        ##update inter-view feature
        if not self.last:            
            fea_c = self.conv_f1(upda_intra_feas)
            out_c = self.conv_f2(torch.cat((fea_c, inter_fea), 1))
        else:
            out_c = inter_fea        
        
        fuse_fea = fuse_fea.unsqueeze(1).contiguous().view(b, -1, c, h, w).permute(0,2,1,3,4)

        return fuse_fea, out_c


class MCB(nn.Module):
    '''
    Multi-view Contex Block
    '''
    def __init__(self, channels, angRes):
        super(MCB, self).__init__()
        self.prelu1 = nn.LeakyReLU(0.02, inplace=True)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.ASPP = D3ResASPP(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

    def forward(self, x_init):
        b, c, n, h, w = x_init.shape
        x = self.conv1(x_init)
        buffer = self.prelu1(x)
        buffer = self.ASPP(buffer)
        x = self.conv2(buffer)+x_init
        #x = self.prelu2(x)
        return x#.permute(0,2,1,3,4)


class RB(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class SELayer(nn.Module):
    '''
    Channel Attention
    '''
    def __init__(self, out_ch,g=16):
        super(SELayer, self).__init__()
        self.att_c = nn.Sequential(
                nn.Conv2d(out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )

    def forward(self,fm):
        ##channel
        fm_pool = F.adaptive_avg_pool2d(fm, (1, 1))
        att = self.att_c(fm_pool)
        fm = fm * att
        return fm


class FBM(nn.Module):
    '''
    Feature Blending 
    '''
    def __init__(self, channel):
        super(FBM, self).__init__()
        self.FERB_1 = RB(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RB(channel)
        self.FERB_4 = RB(channel)
        self.att1 = SELayer(channel)
        self.att2 = SELayer(channel)
        self.att3 = SELayer(channel)
        self.att4 = SELayer(channel)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer_init = x.contiguous().view(b*n, -1, h, w)
        buffer_1 = self.att1(self.FERB_1(buffer_init))
        buffer_2 = self.att2(self.FERB_2(buffer_1))
        buffer_3 = self.att3(self.FERB_3(buffer_2))
        buffer_4 = self.att4(self.FERB_4(buffer_3))
        buffer = buffer_4.contiguous().view(b, n, -1, h, w)
        return buffer

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
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


class D3ResASPP(nn.Module):
    def __init__(self, channel):
        super(D3ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1,1,1), bias=False), 
                                              nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1), dilation=(2,1,1), bias=False), 
                                              nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(4, 1, 1), dilation=(4,1,1), bias=False), 
                                              nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv3d(channel*3, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1,1,1))

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
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out

