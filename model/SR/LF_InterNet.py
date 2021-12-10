'''
@inproceedings{LF_InterNet,
  title={Spatial-angular interaction for light field image super-resolution},
  author={Wang, Yingqian and Wang, Longguang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
  booktitle={European Conference on Computer Vision},
  pages={290--308},
  year={2020},
  organization={Springer}
}
'''
import torch
import torch.nn as nn


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        channels = 64
        self.factor = args.scale_factor
        n_groups, n_blocks = 4, 4

        # Feature Extraction
        self.AngFE = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=int(self.angRes), stride=int(self.angRes), padding=0, bias=False))
        self.SpaFE = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=int(self.angRes), padding=int(self.angRes), bias=False))
        # Spatial-Angular Interaction
        self.CascadeInterBlock = CascadeInterBlock(self.angRes, n_groups, n_blocks, channels)
        # Fusion and Reconstruction
        self.BottleNeck = BottleNeck(self.angRes, n_blocks, channels)
        self.ReconBlock = ReconBlock(self.angRes, channels, self.factor)

    def forward(self, x, Lr_info=None):
        x = SAI2MacPI(x, self.angRes)
        xa = self.AngFE(x)
        xs = self.SpaFE(x)
        buffer_a, buffer_s = self.CascadeInterBlock(xa, xs)
        buffer_out = self.BottleNeck(buffer_a, buffer_s) + xs
        out = self.ReconBlock(buffer_out)
        return out


class make_chains(nn.Module):
    def __init__(self, angRes, channels):
        super(make_chains, self).__init__()

        self.Spa2Ang = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes*angRes*channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.AngConvSq = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaConvSq = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                            padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        buffer_ang1 = xa
        buffer_ang2 = self.ReLU(self.Spa2Ang(xs))
        buffer_spa1 = xs
        buffer_spa2 = self.Ang2Spa(xa)
        buffer_a = torch.cat((buffer_ang1, buffer_ang2), 1)
        buffer_s = torch.cat((buffer_spa1, buffer_spa2), 1)
        out_a = self.ReLU(self.AngConvSq(buffer_a)) + xa
        out_s = self.ReLU(self.SpaConvSq(buffer_s)) + xs
        return out_a, out_s


class InterBlock(nn.Module):
    def __init__(self, angRes, n_layers, channels):
        super(InterBlock, self).__init__()
        modules = []
        self.n_layers = n_layers
        for i in range(n_layers):
            modules.append(make_chains(angRes, channels))
        self.chained_layers = nn.Sequential(*modules)

    def forward(self, xa, xs):
        buffer_a = xa
        buffer_s = xs
        for i in range(self.n_layers):
            buffer_a, buffer_s = self.chained_layers[i](buffer_a, buffer_s)
        out_a = buffer_a
        out_s = buffer_s
        return out_a, out_s


class CascadeInterBlock(nn.Module):
    def __init__(self, angRes, n_blocks, n_layers, channels):
        super(CascadeInterBlock, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(InterBlock(angRes, n_layers, channels))
        self.body = nn.Sequential(*body)
    def forward(self, buffer_a, buffer_s):
        out_a = []
        out_s = []
        for i in range(self.n_blocks):
            buffer_a, buffer_s = self.body[i](buffer_a, buffer_s)
            out_a.append(buffer_a)
            out_s.append(buffer_s)
        return torch.cat(out_a, 1), torch.cat(out_s, 1)


class BottleNeck(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck, self).__init__()

        self.AngBottle = nn.Conv2d(n_blocks*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.SpaBottle = nn.Conv2d((n_blocks+1)*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                    padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, self.Ang2Spa(xa)), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out


class ReconBlock(nn.Module):
    def __init__(self, angRes, channels, upscale_factor):
        super(ReconBlock, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels * upscale_factor ** 2, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor)
        self.FinalConv = nn.Conv2d(int(channels), 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.angRes = angRes

    def forward(self, x):
        buffer = self.PreConv(x)
        bufferSAI_LR = MacPI2SAI(buffer, self.angRes)
        bufferSAI_HR = self.PixelShuffle(bufferSAI_LR)
        out = self.FinalConv(bufferSAI_HR)
        return out


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


def weights_init(m):

    pass





class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss


