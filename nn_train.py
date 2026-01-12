# ..\Venv\Scripts\activate
# run: torchrun --nproc_per_node=1 --master_port 39985 nn_train.py
import datetime
import os
from Forecasting.config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
# from models.encoder_decoder import Encoder_Decoder
from Forecasting.models.attetion_unet import AttU_Net
from Forecasting.models.SDE_HNN import SDEHNN, SDEHNN_1d
from Forecasting.loss import Loss_MSE, GaussianNLLLoss
from Forecasting.loader import Data, Data_1d
import argparse
from collections import OrderedDict
from Forecasting.utils import *
from Forecasting.train import train


if __name__ == '__main__':
    fix_random(2025)

    mask = load_mask(cfg.root_path)
    print(f'CUDA is availiable: {torch.cuda.is_available()}')
    weights = [1, cfg.A_coeff_weight, cfg.B_coeff_weight]

    LR = cfg.LR
    parser = argparse.ArgumentParser()

    # parallel group
    torch.distributed.init_process_group(backend="gloo")
    # threads = cfg.dataloader_thread

    # model = AttU_Net(cfg.in_len * cfg.channels, cfg.out_len * cfg.channels, (cfg.batch, cfg.height, cfg.width), 3, 1, 0)
    model = SDEHNN_1d(1)

    # # optimizer
    # if cfg.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # elif cfg.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # else:
    #     optimizer = None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # loss
    criterion = Loss_MSE().cuda()
    # criterion = GaussianNLLLoss().cuda()

    model_load_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH + f'/models/days_{cfg.out_len}_features_{cfg.features_amount}.pth'
    model_save_path = model_load_path
    # model_load_path = model_save_path
    if cfg.nn_mode == 'test':
        model_load_path = model_save_path
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    # reading model weights if save exists
    print(f'Trying to read {model_load_path},\n exists = {os.path.exists(model_load_path)}\n')
    # logs_file.write(f'Trying to read {model_load_path},\n exists = {os.path.exists(model_load_path)}\n')
    if cfg.LOAD_MODEL and os.path.exists(model_load_path):
        print('Loading model')
        # original saved file with DataParallel
        state_dict = torch.load(model_load_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            if not ('total' in k):
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)

    # create train and test dataloaders, train from 01.01.1979 to 01.01.2024, test from 01.01.2024 to 28.11.2024
    days_delta1 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    # train_data = Data(cfg, 0, days_delta1, weights)
    # test_data = Data(cfg, days_delta1, days_delta1 - cfg.in_len - cfg.out_len + days_delta2, weights)

    train_data = Data_1d(cfg, 0, days_delta1, (40, 40))
    test_data = Data_1d(cfg, days_delta1, days_delta1 - cfg.in_len - cfg.out_len + days_delta2, (40, 40))
    train(train_data, model, criterion, optimizer, mask, model_save_path)


