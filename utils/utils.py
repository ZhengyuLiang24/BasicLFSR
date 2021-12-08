import numpy as np
import os
from skimage import metrics
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from option import args
from einops import rearrange
import xlwt


class ExcelFile():
    def __init__(self):
        self.xlsx_file = xlwt.Workbook()
        self.worksheet = self.xlsx_file.add_sheet(r'sheet1', cell_overwrite_ok=True)
        self.worksheet.write(0, 0, 'Datasets')
        self.worksheet.write(0, 1, 'Scenes')
        self.worksheet.write(0, 2, 'PSNR')
        self.worksheet.write(0, 3, 'SSIM')
        self.worksheet.col(0).width = 256 * 16
        self.worksheet.col(1).width = 256 * 22
        self.worksheet.col(2).width = 256 * 10
        self.worksheet.col(3).width = 256 * 10
        self.sum = 1

    def write_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        for i in range(len(psnr_iter_test)):
            self.add_sheet(test_name, LF_name[i], psnr_iter_test[i], ssim_iter_test[i])

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        self.add_sheet(test_name, 'average', psnr_epoch_test, ssim_epoch_test)
        self.sum = self.sum + 1

    def add_sheet(self, test_name, LF_name, psnr_iter_test, ssim_iter_test):
        ''' Save PSNR & SSIM '''
        self.worksheet.write(self.sum, 0, test_name)
        self.worksheet.write(self.sum, 1, LF_name)
        self.worksheet.write(self.sum, 2, '%.6f' % psnr_iter_test)
        self.worksheet.write(self.sum, 3, '%.6f' % ssim_iter_test)
        self.sum = self.sum + 1


def get_logger(log_dir, args):
    '''LOG '''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def create_dir(args):
    log_dir = Path(args.path_log)
    log_dir.mkdir(exist_ok=True)
    if args.task == 'SR':
        task_path = 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + str(args.scale_factor) + 'x'
    
    log_dir = log_dir.joinpath(task_path)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.data_name)
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(args.model_name)
    log_dir.mkdir(exist_ok=True)

    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = log_dir.joinpath('results/')
    results_dir.mkdir(exist_ok=True)

    return log_dir, checkpoints_dir, results_dir


class Logger():
    def __init__(self, log_dir, args):
        self.logger = get_logger(log_dir, args)

    def log_string(self, str):
        if args.local_rank <= 0:
            self.logger.info(str)
            print(str)


def cal_metrics(args, label, out,):
    if len(label.size()) == 4:
        label = rearrange(label, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)
        out = rearrange(out, 'b c (a1 h) (a2 w) -> b c a1 h a2 w', a1=args.angRes_in, a2=args.angRes_in)

    if len(label.size()) == 5:
        label = label.permute((0, 1, 3, 2, 4)).unsqueeze(0)
        out = out.permute((0, 1, 3, 2, 4)).unsqueeze(0)

    B, C, U, h, V, w = label.size()
    label_y = label[:, 0, :, :, :, :].data.cpu()
    out_y = out[:, 0, :, :, :, :].data.cpu()

    PSNR = np.zeros(shape=(B, U, V), dtype='float32')
    SSIM = np.zeros(shape=(B, U, V), dtype='float32')
    for b in range(B):
        for u in range(U):
            for v in range(V):
                PSNR[b, u, v] = metrics.peak_signal_noise_ratio(label_y[b, u, :, v, :].numpy(), out_y[b, u, :, v, :].numpy())
                if args.task == 'RE':
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True,
                                                                  sigma=1.5, use_sample_covariance=False)
                else:
                    SSIM[b, u, v] = metrics.structural_similarity(label_y[b, u, :, v, :].numpy(),
                                                                  out_y[b, u, :, v, :].numpy(),
                                                                  gaussian_weights=True, )
                pass

    if args.task=='RE':
        for u in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
            for v in range(0, args.angRes_out, (args.angRes_out - 1) // (args.angRes_in - 1)):
                PSNR[:, u, v] = 0
                SSIM[:, u, v] = 0

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean



def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    # H, W = numU*pH, numV*pW
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]
            #plt.figure()
            #plt.imshow(outLF_temp)
            outLF[u, v, :, :] = temp[0:h0, 0:w0]
    #plt.figure()
    #plt.imshow(outLF)

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y
