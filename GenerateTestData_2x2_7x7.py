import argparse
import os
import h5py
from utils.imresize import *
from pathlib import Path
import scipy.io as scio
import sys
from utils.utils import rgb2ycbcr
from einops import rearrange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes_in", type=int, default=2, help="angular resolution")
    parser.add_argument("--angRes_out", type=int, default=7, help="angular resolution")
    parser.add_argument('--data_for', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default='./datasets_asr/', help='')
    parser.add_argument('--save_data_path', type=str, default='./', help='')

    return parser.parse_args()


def main(args):
    angRes_in, angRes_out = args.angRes_in, args.angRes_out

    ''' dir '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath('RE_' + str(angRes_in) + 'x' + str(angRes_in) + '_' + str(angRes_out) + 'x' + str(angRes_out))
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['HCI_new', "HCI_old", "30scene", "occlusions", "reflective"]:
        # if src_datasets[index_dataset] not in ['30scene']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating test data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(root + file)
                    LF = np.array(data['LF'])

                LF = LF[1:8, 1:8, :, :, :3]
                (U, V, H, W, _) = LF.shape
                H = H // 4 * 4
                W = W // 4 * 4

                # Extract central angRes * angRes views
                LF = LF[0:angRes_out, 0:angRes_out, 0:H, 0:W, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape

                idx_save = idx_save + 1
                idx_scene_save = idx_scene_save + 1
                Hr_SAI_y = np.zeros((angRes_out * H, angRes_out * W), dtype='single')
                Sr_SAI_cbcr = np.zeros((angRes_out * H, angRes_out * W, 2), dtype='single')
                Lr_SAI_y = np.zeros((angRes_in * H, angRes_in * W), dtype='single')
                Lr_SAI_cb = np.zeros((angRes_in * H, angRes_in * W), dtype='single')
                Lr_SAI_cr = np.zeros((angRes_in * H, angRes_in * W), dtype='single')
                for u in range(0, U, 6):
                    for v in range(0, V, 6):
                        tmp_Hr_rgb = LF[u, v, :, :, :]
                        tmp_Hr_ycbcr = rgb2ycbcr(tmp_Hr_rgb)
                        u0 = u // 6
                        v0 = v // 6
                        Lr_SAI_y[u0 * H: (u0 + 1) * H, v0 * W: (v0 + 1) * W] = tmp_Hr_ycbcr[:, :, 0]
                        Lr_SAI_cb[u0 * H: (u0 + 1) * H, v0 * W: (v0 + 1) * W] = tmp_Hr_ycbcr[:, :, 1]
                        Lr_SAI_cr[u0 * H: (u0 + 1) * H, v0 * W: (v0 + 1) * W] = tmp_Hr_ycbcr[:, :, 2]

                Lr_SAI_cb = rearrange(Lr_SAI_cb, "(u h) (v w) -> h w u v", u=2, v=2)
                Lr_SAI_cr = rearrange(Lr_SAI_cr, "(u h) (v w) -> h w u v", u=2, v=2)

                SR_cb = np.zeros((H, W, angRes_out, angRes_out), dtype='single')
                SR_cr = np.zeros((H, W, angRes_out, angRes_out), dtype='single')
                for h in range(Lr_SAI_cb.shape[0]):
                    for w in range(Lr_SAI_cb.shape[1]):
                            SR_cb[h, w, :, :] = imresize(Lr_SAI_cb[h, w, :, :], angRes_out / angRes_in)
                            SR_cr[h, w, :, :] = imresize(Lr_SAI_cr[h, w, :, :], angRes_out / angRes_in)


                SR_cb = rearrange(SR_cb, "h w u v -> (u h) (v w)", u=7, v=7)
                SR_cr = rearrange(SR_cr, "h w u v -> (u h) (v w)", u=7, v=7)
                Sr_SAI_cbcr[:, :, 0] = SR_cb
                Sr_SAI_cbcr[:, :, 1] = SR_cr


                for u in range(U):
                    for v in range(V):
                        tmp_Hr_rgb = LF[u, v, :, :, :]
                        tmp_Hr_ycbcr = rgb2ycbcr(tmp_Hr_rgb)
                        Hr_SAI_y[u * H: (u+1) * H, v * W: (v+1)* W] = tmp_Hr_ycbcr[:, :, 0]
                        pass
                    pass

                file_name = [str(sub_save_dir) + '/' + '%s' % file.split('.')[0] + '.h5']
                with h5py.File(file_name[0], 'w') as hf:
                    hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
                    hf.create_dataset('Sr_SAI_cbcr', data=Sr_SAI_cbcr.transpose((2, 1, 0)), dtype='single')
                    hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')
                    hf.close()
                    pass

                print('%d test samples have been generated\n' % (idx_scene_save))
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    args = parse_args()

    main(args)
