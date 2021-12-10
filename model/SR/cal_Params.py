import importlib
import torch
from option import args


if __name__ == "__main__":
    args.model_name = 'EDSR'
    args.angRes_in = 5
    args.scale_factor = 4

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    MODEL_PATH = args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of parameters: %.4fM' % (total / 1e6))
