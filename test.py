import importlib
import torch
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import MultiTestSetDataLoader
from collections import OrderedDict
from train import test


def main(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()

        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
            excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

            psnr_epoch_test = float(np.array(psnr_iter_test).mean())
            ssim_epoch_test = float(np.array(ssim_iter_test).mean())
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print('Test on %s, psnr/ssim is %.3f/%.4f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass

        psnr_mean_test = float(np.array(psnr_testset).mean())
        ssim_mean_test = float(np.array(ssim_testset).mean())
        excel_file.add_sheet('ALL', 'Average', psnr_mean_test, ssim_mean_test)
        print('The mean psnr on testsets is %.5f, mean ssim is %.5f' % (psnr_mean_test, ssim_mean_test))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xls')

    pass


if __name__ == '__main__':
    from option import args

    main(args)
