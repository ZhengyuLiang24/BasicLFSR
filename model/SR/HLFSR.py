import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class get_model(nn.Module):
	def __init__(self, args):
		super(get_model, self).__init__()
		n_blocks = 15
		channels = 64
		self.angRes = args.angRes_in
		self.upscale_factor = args.scale_factor

		self.HFEM_1 = HFEM(self.angRes, n_blocks, channels, first=True)
		self.HFEM_2 = HFEM(self.angRes, n_blocks, channels, first=False)
		self.HFEM_3 = HFEM(self.angRes, n_blocks, channels, first=False)
		self.HFEM_4 = HFEM(self.angRes, n_blocks, channels, first=False)
		self.HFEM_5 = HFEM(self.angRes, n_blocks, channels, first=False)

		# define tail module for upsamling
		UpSample = [
			Upsampler(self.upscale_factor, channels, kernel_size=3, stride=1, dilation=1, padding=1, act=False),
			nn.Conv2d(channels, 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=False)]
		self.UpSample = nn.Sequential(*UpSample)

	def forward(self, x, info=None):
		# Upscaling
		x_upscale = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

		# Reshaping
		x = SAI2MacPI(x, self.angRes)
		HFEM_1 = self.HFEM_1(x)
		HFEM_2 = self.HFEM_2(HFEM_1)
		HFEM_3 = self.HFEM_3(HFEM_2)
		HFEM_4 = self.HFEM_4(HFEM_3)
		HFEM_5 = self.HFEM_5(HFEM_4)

		# Reshaping
		x_out = MacPI2SAI(HFEM_5, self.angRes)
		x_out = self.UpSample(x_out)
		x_out += x_upscale
		return x_out


class HFEM(nn.Module):
	def __init__(self, angRes, n_blocks, channels, first=False):
		super(HFEM, self).__init__()
		self.first = first 
		self.n_blocks = n_blocks
		self.angRes = angRes

		# define head module epi feature
		head_epi = []
		if first:  
			head_epi.append(nn.Conv2d(angRes, channels, kernel_size=3, stride=1, padding=1, bias=False))
		else:
			head_epi.append(nn.Conv2d(angRes*channels, channels, kernel_size=3, stride=1, padding=1, bias=False))

		self.head_epi = nn.Sequential(*head_epi)

		self.epi2spa = nn.Sequential(
			nn.Conv2d(4*channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes),
		)

		# define head module intra spatial feature
		head_spa_intra = []
		if first:  
			head_spa_intra.append(nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=int(angRes),
											padding=int(angRes), bias=False))
		else:
			head_spa_intra.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
											padding=int(angRes), bias=False))

		self.head_spa_intra = nn.Sequential(*head_spa_intra)

		# define head module inter spatial feature
		head_spa_inter = []
		if first:  
			head_spa_inter.append(nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False))
		else:
			head_spa_inter.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False))

		self.head_spa_inter = nn.Sequential(*head_spa_inter)

		# define head module intra angular feature
		head_ang_intra = []
		if first: 
			head_ang_intra.append(nn.Conv2d(1, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
											padding=0, bias=False))

		else:
			head_ang_intra.append(nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
											padding=0, bias=False))

		self.head_ang_intra = nn.Sequential(*head_ang_intra)

		self.ang2spa_intra = nn.Sequential(
			nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(angRes), 
		)

		# define head module inter angular feature
		head_ang_inter = []
		if first:  
			head_ang_inter.append(nn.Conv2d(1, channels, kernel_size=int(angRes*2), stride=int(angRes*2), dilation=1,
											padding=0, bias=False))

		else:
			head_ang_inter.append(nn.Conv2d(channels, channels, kernel_size=int(angRes*2), stride=int(angRes*2),
											dilation=1, padding=0, bias=False))

		self.head_ang_inter = nn.Sequential(*head_ang_inter)

		self.ang2spa_inter = nn.Sequential(
			nn.Conv2d(channels, int(4*angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
			nn.PixelShuffle(2*angRes),
		)

		# define  module attention fusion feature
		self.attention_fusion = AttentionFusion(channels)
											
		# define  module spatial residual group
		self.SRG = nn.Sequential(
			nn.Conv2d(5*channels, channels, kernel_size=1, stride =1, dilation=1, padding=0, bias=False),
			ResidualGroup(self.n_blocks, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes))
		)

	def forward(self, x):
		# MO-EPI feature extractor
		data_0, data_90, data_45, data_135 = MacPI2EPI(x, self.angRes)

		data_0 = self.head_epi(data_0)
		data_90 = self.head_epi(data_90)
		data_45 = self.head_epi(data_45)
		data_135 = self.head_epi(data_135)
	
		mid_merged = torch.cat((data_0, data_90, data_45, data_135), 1)
		x_epi = self.epi2spa(mid_merged)

		# intra/inter spatial feature extractor
		x_s_intra = self.head_spa_intra(x)
		x_s_inter = self.head_spa_inter(x)
	
		# intra/inter angular feature extractor
		x_a_intra = self.head_ang_intra(x)
		x_a_intra = self.ang2spa_intra(x_a_intra)

		x_a_inter = self.head_ang_inter(x)
		x_a_inter = self.ang2spa_inter(x_a_inter)

		# fusion feature and refinement
		out = x_s_intra.unsqueeze(1)
		out = torch.cat([x_s_inter.unsqueeze(1), out], 1)
		out = torch.cat([x_a_intra.unsqueeze(1), out], 1)
		out = torch.cat([x_a_inter.unsqueeze(1), out], 1)
		out = torch.cat([x_epi.unsqueeze(1), out], 1)

		[out, att_weight] = self.attention_fusion(out)
		out = self.SRG(out)
		return out


class AttentionFusion(nn.Module):
	def __init__(self, channels, eps=1e-5):
		super(AttentionFusion, self).__init__()
		self.epsilon = eps
		self.alpha = nn.Parameter(torch.ones(1))
		self.gamma = nn.Parameter(torch.zeros(1))
		self.beta = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		m_batchsize, N, C, height, width = x.size()
		x_reshape = x.view(m_batchsize, N, -1)
		M = C * height * width

		# compute covariance feature
		mean = torch.mean(x_reshape, dim=-1).unsqueeze(-1)
		x_reshape = x_reshape - mean
		cov = (1 / (M - 1) * x_reshape @ x_reshape.transpose(-1, -2)) * self.alpha
		# print(cov)
		norm = cov / ((cov.pow(2).mean((1, 2), keepdim=True) + self.epsilon).pow(0.5))  # l-2 norm

		attention = torch.tanh(self.gamma * norm + self.beta)
		x_reshape = x.view(m_batchsize, N, -1)

		out = torch.bmm(attention, x_reshape)
		out = out.view(m_batchsize, N, C, height, width)

		out += x
		out = out.view(m_batchsize, -1, height, width)
		return out, attention


## Residual Channel Attention Block (RCAB)
class ResidualBlock(nn.Module):
	def __init__(self, n_feat, kernel_size, stride, dilation, padding):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
		self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
		self.relu = nn.ReLU(inplace=True)
		# # initialization
		# initialize_weights([self.conv1, self.conv2], 0.1)
		self.CALayer = CALayer(n_feat, reduction=int(n_feat//4))

	def forward(self, x):
		out = self.relu(self.conv1(x))
		out = self.conv2(out)
		out = self.CALayer(out)
		return x + out


## Residual Group
class ResidualGroup(nn.Module):
	def __init__(self, n_blocks, n_feat, kernel_size, stride, dilation, padding):
		super(ResidualGroup, self).__init__()
		self.fea_resblock = make_layer(ResidualBlock, n_feat, n_blocks,kernel_size, stride, dilation, padding)
		self.conv = nn.Conv2d(n_feat, n_feat,  kernel_size=kernel_size, stride=stride, dilation=dilation,
							  padding=padding, bias=True)

	def forward(self, x):
		res = self.fea_resblock(x)
		res = self.conv(res)
		res += x
		return res


def make_layer(block, nf, n_layers,kernel_size, stride, dilation, padding ):
	layers = []
	for _ in range(n_layers):
		layers.append(block(nf, kernel_size, stride, dilation, padding))
	return nn.Sequential(*layers)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CALayer, self).__init__()
		# global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# feature channel downscale and upscale --> channel weight
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel // reduction, 1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // reduction, channel, 1, padding=0),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y


class Upsampler(nn.Sequential):
	def __init__(self, scale, n_feat,kernel_size, stride, dilation, padding,  bn=False, act=False, bias=True):

		m = []
		if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
			for _ in range(int(math.log(scale, 2))):
				m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
				m.append(nn.PixelShuffle(2))
				if bn: m.append(nn.BatchNorm2d(n_feat))
				if act: m.append(act())
		elif scale == 3:
			m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
			m.append(nn.PixelShuffle(3))
			if bn: m.append(nn.BatchNorm2d(n_feat))
			if act: m.append(act())
		else:
			raise NotImplementedError

		super(Upsampler, self).__init__(*m)


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


def SAI24DLF(x, angRes):
	uh, vw = x.shape
	h0, w0 = int(uh // angRes), int(vw // angRes)

	LFout = torch.zeros(angRes, angRes, h0, w0)
	for u in range(angRes):
		start_u = u * h0
		end_u = (u + 1) * h0
		for v in range(angRes):
			start_v = v * w0
			end_v = (v + 1) * w0
			img_tmp = x[start_u:end_u, start_v:end_v]
			LFout[u, v, :, :] = img_tmp

	return LFout


def MacPI2SAI(x, angRes):
	out = []
	for i in range(angRes):
		out_h = []
		for j in range(angRes):
			out_h.append(x[:, :, i::angRes, j::angRes])
		out.append(torch.cat(out_h, 3))
	out = torch.cat(out, 2)
	return out


def MacPI2EPI(x, angRes):
	data_0 = []
	data_90 = []
	data_45 = []
	data_135 = []

	index_center = int(angRes // 2)
	for i in range(0, angRes, 1):
		img_tmp = x[:, :, index_center::angRes, i::angRes]
		data_0.append(img_tmp)
	data_0 = torch.cat(data_0, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:, :, i::angRes, index_center::angRes]
		data_90.append(img_tmp)
	data_90 = torch.cat(data_90, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:, :, i::angRes, i::angRes]
		data_45.append(img_tmp)
	data_45 = torch.cat(data_45, 1)

	for i in range(0, angRes, 1):
		img_tmp = x[:, :, i::angRes, angRes - i - 1::angRes]
		data_135.append(img_tmp)
	data_135 = torch.cat(data_135, 1)

	return data_0, data_90, data_45, data_135