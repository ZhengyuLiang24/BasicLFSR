import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        n_blocks, channel = 5, 36
        angRes = args.angRes_in
        factor = args.scale_factor
        self.angRes = angRes
        self.factor = factor
        self.get_gradient = Get_gradient()
        self.srbranch = ContentBranch(angRes,factor)
        self.gbranch = GradientBranch(angRes,factor) # GradientBranch is same as ContentBranch
        self.fuse = FusionTransformer()
        self.Reconstruct = CascadedBlocks(n_blocks, 3* channel)
        self.UpSample = Upsample(3,channel, factor)

    def forward(self, x, Lr_Info):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
        g = self.get_gradient(x)
        s = self.srbranch(x)
        d = self.gbranch(g)
        fuse_feature = self.fuse(s,d)
        fuse_feature = self.Reconstruct(fuse_feature)
        out = self.UpSample(fuse_feature)
        out = FormOutput(out) + x_upscale
        return out


class SALSA(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=4, softmax_scale=10, shape=64, p_len=64, in_channels=64,
                 inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False):
        super(SALSA, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE
        self.conv33 = nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, b):
        output = []
        for B in range(b.shape[0]):
            b_one = b[B]
            kernel = self.ksize

            # spatial-angular convolutional tokenization
            b1 = self.g(b_one)
            b2 = self.theta(b_one)
            b3 = self.phi(b_one)

            raw_int_bs = list(b1.size())

            patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                          strides=[self.stride_1, self.stride_1],
                                                          rates=[1, 1],
                                                          padding='same')
            patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_28 = patch_28.permute(0, 4, 1, 2, 3)

            patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                            strides=[self.stride_2, self.stride_2],
                                                            rates=[1, 1],
                                                            padding='same')
            patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_112 = patch_112.permute(0, 4, 1, 2, 3)

            patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                                strides=[self.stride_2, self.stride_2],
                                                                rates=[1, 1],
                                                                padding='same')
            patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)

            _, paddings = same_padding(torch.split(b3, 1, dim=0)[0], [self.ksize, self.ksize], [1, 1], [1, 1])

            # spatial-angular self-attention
            q = patch_28.contiguous().view(patch_28.shape[0] * patch_28.shape[1], -1)
            k = patch_112_2.permute(2, 3, 4, 0, 1)
            k = k.contiguous().view(-1, k.shape[3] * k.shape[4])
            score_map = torch.matmul(q, k)

            b_s, l_s, h_s, w_s = b_one.shape[0], patch_28.shape[1], b_one.shape[2], b_one.shape[3]
            att = F.softmax(score_map * self.softmax_scale, dim=1)
            v = patch_112.contiguous().view(patch_112.shape[0] * patch_112.shape[1], -1)
            attMulV = torch.mm(att, v)

            zi = attMulV.view(b_s, l_s, -1).permute(0, 2, 1)


            # spatial-angular convolutional de-tokenization
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0],
                                          stride=self.stride_1)

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel),
                                                padding=paddings[0], stride=self.stride_1)

            zi = zi / out_mask

            y = self.W(zi)
            y = b_one + y
            if self.add_SE:
                y_SE = self.SE(y)
                y = self.conv33(torch.cat((y_SE * y, y), dim=1))
            output.append(y)
        output = torch.stack(output, dim=0)
        return output

    def GSmap(self, a, b):
        return torch.matmul(a, b)


class CrossAttentionSALSA(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=4, softmax_scale=10, shape=64, p_len=64, in_channels=64
                 , inter_channels=16, use_multiple_size=False, use_topk=False, add_SE=False):
        super(CrossAttentionSALSA, self).__init__()
        self.ksize = ksize
        self.shape = shape
        self.p_len = p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size = use_multiple_size
        self.use_topk = use_topk
        self.add_SE = add_SE
        self.conv33 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, s,g):
        # s content
        # g gradient
        output = []
        for B in range(s.shape[0]):  # for each batch
            b_one = s[B]
            d_one = g[B]
            kernel = self.ksize

            # spatial-angular convolutional tokenization
            b1 = self.g(b_one)
            b2 = self.theta(d_one)
            b3 = self.phi(d_one)

            raw_int_bs = list(b1.size())

            patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                          strides=[self.stride_1, self.stride_1],
                                                          rates=[1, 1],
                                                          padding='same')
            patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_28 = patch_28.permute(0, 4, 1, 2, 3)

            patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                            strides=[self.stride_2, self.stride_2],
                                                            rates=[1, 1],
                                                            padding='same')
            patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_112 = patch_112.permute(0, 4, 1, 2, 3)

            patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                                strides=[self.stride_2, self.stride_2],
                                                                rates=[1, 1],
                                                                padding='same')
            patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)

            _, paddings = same_padding(torch.split(b3, 1, dim=0)[0], [self.ksize, self.ksize], [1, 1],
                                       [1, 1])

            # spatial-angular self-attention
            q = patch_28.contiguous().view(patch_28.shape[0] * patch_28.shape[1], -1)
            k = patch_112_2.permute(2, 3, 4, 0, 1)
            k = k.contiguous().view(-1, k.shape[3] * k.shape[4])
            score_map = torch.matmul(q, k)

            b_s, l_s, h_s, w_s = b_one.shape[0], patch_28.shape[1], b_one.shape[2], b_one.shape[3]
            att = F.softmax(score_map * self.softmax_scale, dim=1)
            v = patch_112.contiguous().view(patch_112.shape[0] * patch_112.shape[1], -1)
            attMulV = torch.mm(att, v)

            zi = attMulV.view(b_s, l_s, -1).permute(0, 2, 1)

            # spatial-angular convolutional de-tokenization
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0],
                                          stride=self.stride_1)

            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel),
                                                padding=paddings[0], stride=self.stride_1)

            zi = zi / out_mask

            y = self.W(zi)
            y = b_one + y
            if self.add_SE:
                y_SE = self.SE(y)
                y = self.conv33(torch.cat((y_SE * y, y), dim=1))
            output.append(y)
        output = torch.stack(output, dim=0)
        return output

    def GSmap(self, a, b):
        return torch.matmul(a, b)


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        self.kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        self.kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        self.weight_h = nn.Parameter(data=self.kernel_h, requires_grad=False).to(x.device)
        self.weight_v = nn.Parameter(data=self.kernel_v, requires_grad=False).to(x.device)
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)
        x = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        return x


class ADAM(nn.Module):
    def __init__(self, channel, angRes):
        super(ADAM, self).__init__()
        self.conv_1 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        self.ASPP = ResASPP(channel)
        self.conv_f1 = nn.Conv2d(angRes*angRes*channel, angRes*angRes*channel, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x_cv = x[:,12,:,:,:]
        x_sv_part1 = x[:,0:12,:,:,:]
        x_sv_part2 = x[:, 13:25, :, :, :]
        x_sv = torch.cat([x_sv_part1,x_sv_part2],dim=1)

        b, n, c, h, w = x_sv.shape
        aligned_fea = []
        for i in range(n):
            current_sv = x_sv[:, i, :, :, :].contiguous()
            buffer = torch.cat((current_sv, x_cv), dim=1)           # B * 2C * H * W
            buffer = self.lrelu(self.conv_1(buffer))
            buffer = self.ASPP(buffer)
            aligned_fea.append(buffer)
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
            fuse_sv = torch.cat((current_sv, buffer), dim=1)
            fuse_sv = self.conv_f3(fuse_sv)
            out_sv.append(fuse_sv)
        out_sv = torch.stack(out_sv, dim=1)
        out_cv = self.conv_f3(torch.cat((x_cv, fuse_fea[:, 0, :, :, :]), 1))
        out =FormOutput_ADAM(out_sv,out_cv)

        return out


class salsa(nn.Module):
    def __init__(self, feat_num):
        super(salsa, self).__init__()
        self.attention = SALSA(in_channels=feat_num)

    def forward(self, x):
        x = x + self.attention(x)
        return x


class crossattentionsalsa(nn.Module):
    def __init__(self, feat_num):
        super(crossattentionsalsa, self).__init__()
        self.attention = CrossAttentionSALSA(in_channels=feat_num)

    def forward(self, s,g):
        s = s + self.attention(s,g)
        return s


class FusionTransformer(nn.Module):
    def __init__(self):
        super(FusionTransformer, self).__init__()
        channel = 36
        block =3
        self.trans_f_row = crossattentionsalsa(block*channel)
        self.trans_f_col = crossattentionsalsa(block*channel)
    def forward(self,s,g):
        buffer_row = []
        for i in range(5):
            row_s = s[:, 5 * i:5 * (i + 1)]
            row_d = g[:, 5 * i:5 * (i + 1)]
            Tran_row = self.trans_f_row(row_s,row_d)
            buffer_row.append(Tran_row)
        buffer_row = torch.cat(buffer_row, dim=1)

        buffer_col = []
        for i in range(5):
            col_s = []
            col_g = []
            for j in range(5):
                col_s.append(buffer_row[:, 5 * j + i].unsqueeze(1))
                col_g.append(g[:, 5 * j + i].unsqueeze(1))
            col_s = torch.cat(col_s, dim=1)
            col_g = torch.cat(col_g, dim=1)
            Tran_col = self.trans_f_col(col_s,col_g)
            buffer_col.append(Tran_col)
        buffer_col = torch.cat(buffer_col, dim=1)
        out = Col_T(buffer_col)

        return out


class ContentBranch(nn.Module):
    def __init__(self, angRes, factor):
        super(ContentBranch, self).__init__()
        channel = 36
        self.factor = factor
        self.angRes = angRes
        self.FeaExtract = FeaExtract(channel)
        self.ADAM_1 = ADAM(channel, angRes)

        ## ContentTransformer
        self.trans_row1 = salsa(channel)
        self.trans_col1 = salsa(channel)
        self.trans_row2 = salsa(channel)
        self.trans_col2 = salsa(channel)

    def forward(self, x):
        x = LFsplit(x, self.angRes)

        buffer_0 = self.FeaExtract(x)
        buffer_1 = self.ADAM_1(buffer_0)

        buffer_row = []
        for i in range(5):
            row = buffer_1[:, 5 * i:5 * (i + 1)]
            Tran_row = self.trans_row1(row)
            buffer_row.append(Tran_row)
        buffer_row = torch.cat(buffer_row, dim=1)

        buffer_col = []
        for i in range(5):
            col = []
            for j in range(5):
                col.append(buffer_row[:, 5 * j + i].unsqueeze(1))
            col = torch.cat(col, dim=1)
            Tran_col = self.trans_col1(col)
            buffer_col.append(Tran_col)
        buffer_col = torch.cat(buffer_col, dim=1)
        buffer_1 = Col_T(buffer_col)

        buffer_row = []
        for i in range(5):
            row = buffer_1[:, 5 * i:5 * (i + 1)]
            Tran_row = self.trans_row2(row)
            buffer_row.append(Tran_row)
        buffer_row = torch.cat(buffer_row, dim=1)

        buffer_col = []
        for i in range(5):
            col = []
            for j in range(5):
                col.append(buffer_row[:, 5 * j + i].unsqueeze(1))
            col = torch.cat(col, dim=1)
            Tran_col = self.trans_col2(col)
            buffer_col.append(Tran_col)
        buffer_col = torch.cat(buffer_col, dim=1)
        buffer_2 = Col_T(buffer_col)

        out = torch.cat((buffer_0, buffer_1, buffer_2), dim=2)

        return out


class GradientBranch(nn.Module):
    def __init__(self, angRes, factor):
        super(GradientBranch, self).__init__()
        channel = 36
        self.factor = factor
        self.angRes = angRes
        self.FeaExtract = FeaExtract(channel)
        self.ADAM_1 = ADAM(channel, angRes)

        ## GradientTransformer
        self.trans_row1 = salsa(channel)
        self.trans_col1 = salsa(channel)
        self.trans_row2 = salsa(channel)
        self.trans_col2 = salsa(channel)

    def forward(self, x):
        x = LFsplit(x, self.angRes)

        buffer_0 = self.FeaExtract(x)
        buffer_1 = self.ADAM_1(buffer_0)

        buffer_row = []
        for i in range(5):
            row = buffer_1[:, 5 * i:5 * (i + 1)]
            Tran_row = self.trans_row1(row)
            buffer_row.append(Tran_row)
        buffer_row = torch.cat(buffer_row, dim=1)

        buffer_col = []
        for i in range(5):
            col = []
            for j in range(5):
                col.append(buffer_row[:, 5 * j + i].unsqueeze(1))
            col = torch.cat(col, dim=1)
            Tran_col = self.trans_col1(col)
            buffer_col.append(Tran_col)
        buffer_col = torch.cat(buffer_col, dim=1)
        buffer_1 = Col_T(buffer_col)

        buffer_row = []
        for i in range(5):
            row = buffer_1[:, 5 * i:5 * (i + 1)]
            Tran_row = self.trans_row2(row)
            buffer_row.append(Tran_row)
        buffer_row = torch.cat(buffer_row, dim=1)

        buffer_col = []
        for i in range(5):
            col = []
            for j in range(5):
                col.append(buffer_row[:, 5 * j + i].unsqueeze(1))
            col = torch.cat(col, dim=1)
            Tran_col = self.trans_col2(col)
            buffer_col.append(Tran_col)
        buffer_col = torch.cat(buffer_col, dim=1)
        buffer_2 = Col_T(buffer_col)

        out = torch.cat((buffer_0, buffer_1, buffer_2), dim=2)

        return out


def Col_T(feature):
    feature_T = []
    for i in range(5):
        col = []
        for j in range(5):
            col.append(feature[:, 5 * j + i].unsqueeze(1))
        col = torch.cat(col, dim=1)
        feature_T.append(col)
    feature_T = torch.cat(feature_T, dim=1)
    return feature_T


class Upsample(nn.Module):
    def __init__(self, blocks,channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(blocks* channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
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

    def forward(self, x):
        b, n, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        buffer_x_0 = self.FEconv(x)
        buffer_x = self.FERB_1(buffer_x_0)
        buffer_x = self.FERB_2(buffer_x)
        buffer_x = self.FERB_3(buffer_x)
        buffer_x = self.FERB_4(buffer_x)
        _, c, h, w = buffer_x.shape
        buffer_x = buffer_x.unsqueeze(1).contiguous().view(b, -1, c, h, w)  # buffer_sv:  B, N, C, H, W

        return buffer_x


class ResidualBlocks(nn.Module):
    def __init__(self, n_blocks, channel):
        super(ResidualBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(ResBlock(channel))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.body[i](x)
        return x


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


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer = x.contiguous().view(b * n, -1, h, w)
        buffer = self.conv01(buffer)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        buffer = buffer.contiguous().view(b, n, -1, h, w)
        return buffer + x


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
        self.conv_1 = nn.Conv2d(3 * channel // 4, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(3 * channel // 4, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(3 * channel // 4, channel // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv_t = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer = x.contiguous().view(b * n, -1, h, w)
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
    c = C // 4
    output_1 = input[:, :c, :, :]
    output_2 = input[:, c:, :, :]
    return output_1, output_2


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel * 3, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_out = []
    for u in range(angRes):
        for v in range(angRes):
            data_out.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])
    data_out = torch.cat(data_out, dim=1)
    return data_out


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


def FormOutput_ADAM(x_sv, x_cv):
    x_sv_part1 = x_sv[:, 0:12, :, :, :]
    x_sv_part2 = x_sv[:, 12:24, :, :, :]
    x_cv = x_cv.unsqueeze(1)
    out = torch.cat([x_sv_part1,x_cv,x_sv_part2],dim=1)

    return out


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)

    return images, paddings


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss

