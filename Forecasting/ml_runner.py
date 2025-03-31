from struct import unpack
import datetime
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
# from skimage.metrics import structural_similarity as ssim
# import tensorflow as tf
import sys

# SHORT_POSTFIX = '_short'
SHORT_POSTFIX = ''
files_path_prefix = 'E:/Nastya/Data/OceanFull/'
import copy
from loader import Data2
# from models.encoder_decoder import Encoder_Decoder
from loss import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loader import count_offset, load_mask
import argparse
from utils import *
from utils import normalize_data_cuda
from loader import Data2
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    fix_random(2024)
    mask = load_mask(cfg.root_path)

    LR = cfg.LR
    parser = argparse.ArgumentParser()
    start_year = 1979
    end_year, offset = count_offset(start_year)

    # ---------------------------------------------------------------------------------------
    # configs
    width = 91
    height = 81
    batch_size = cfg.batch
    cfg.in_len = 7
    cfg.out_len = 10
    features_amount = 6


    # parallel group
    torch.distributed.init_process_group(backend="gloo")
    threads = cfg.dataloader_thread


    # ---------------------------------------------------------------------------------------
    # create train and test dataloaders, train from 01.01.1979 to 01.01.2024, test from 01.01.2024 to 28.11.2024
    days_delta1 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days
    days_delta2 = (datetime.datetime(2024, 11, 28, 0, 0) - datetime.datetime(2024, 1, 1, 0, 0)).days
    train_data = Data2(cfg, 0, days_delta1)
    test_data = Data2(cfg, days_delta1, days_delta1 - cfg.in_len - cfg.out_len + days_delta2)

    test_sampler = DistributedSampler(test_data, shuffle=False)
    test_loader = DataLoader(test_data, num_workers=threads, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                             sampler=test_sampler)
    criterion_MSE = Loss_MSE().cuda()

    # with ((torch.no_grad())):
    #     ssim_flux = 0.0
    #     ssim_sst = 0.0
    #     ssim_press = 0.0
    #     mse_flux = 0.0
    #     mse_sst = 0.0
    #     mse_press = 0.0
    #     amount = 0
    #     for idx, test_batch in enumerate(test_loader):
    #         print(f'batch {idx}')
    #         test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)
    #         input = test_batch[:, :cfg.in_len, :3].clone()
    #         # print(input[0])
    #         # print('\n\n')
    #         test_pred_values = torch.zeros((input.shape[0], cfg.out_len, 3, input.shape[3], input.shape[4]))
    #         for i in range(3):
    #             test_pred_values[:, :, i] = torch.mean(input[:, :, i])
    #
    #         loss_simple = criterion_MSE(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
    #                                     test_pred_values[:, :, :3])
    #         prediction = test_pred_values[:, :, :3]
    #         truth = test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3]
    #         truth = truth.cpu().numpy()
    #         prediction = prediction.cpu().numpy()
    #         ssim = get_SSIM(prediction, truth, mask)
    #         ssim_arr = np.mean(ssim, axis=(0, 1))
    #         # print(ssim_arr)
    #         ssim_flux += ssim_arr[0]
    #         ssim_sst += ssim_arr[1]
    #         ssim_press += ssim_arr[2]
    #
    #         mse_arr = (prediction - truth)**2 # b s c h w
    #         mse_arr = np.sum(mse_arr, axis=(0, 1, 3, 4))
    #         mse_flux += mse_arr[0]
    #         mse_sst += mse_arr[1]
    #         mse_press += mse_arr[2]
    #
    #
    #         amount += 1
    #
    # print(f'DUMB SSIM flux = {ssim_flux / amount:.4f}')
    # print(f'DUMB SSIM sst = {ssim_sst / amount:.4f}')
    # print(f'DUMB SSIM press = {ssim_press / amount:.4f}')
    # print(f'DUMB MSE flux = {mse_flux / amount:.4f}')
    # print(f'DUMB MSE sst = {mse_sst / amount:.4f}')
    # print(f'DUMB MSE press = {mse_press / amount:.4f}')


    with ((torch.no_grad())):
        ssim_flux = 0.0
        ssim_sst = 0.0
        ssim_press = 0.0
        mse_flux = 0.0
        mse_sst = 0.0
        mse_press = 0.0
        amount = 0
        for idx, test_batch in enumerate(test_loader):
            print(f'batch {idx}')
            test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)
            input = test_batch[:, :cfg.in_len, :3].clone()
            # print(input[0])
            # print('\n\n')
            test_pred_values = torch.zeros((input.shape[0], cfg.out_len, 3, input.shape[3], input.shape[4]))
            t_in = [m for m in range(1, cfg.in_len + 1)]
            t_out = [m for m in range(cfg.in_len + 1, cfg.in_len + 1 + cfg.out_len)]
            for b in range(test_batch.shape[0]):
                for i in range(3):
                    x = input[b, :, i].cpu().numpy()
                    # print(x.shape)
                    for p1 in range(81):
                        for p2 in range(91):
                            fit = np.polyfit(t_in, x[:, p1, p2], 3)
                            line = np.poly1d(fit)
                            test_pred_values[b, :, i, p1, p2] = torch.tensor(line(t_out))

            loss_simple = criterion_MSE(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3],
                                        test_pred_values[:, :, :3], mask)
            prediction = test_pred_values[:, :, :3]
            truth = test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :3]
            truth = truth.cpu().numpy()
            prediction = prediction.cpu().numpy()
            ssim = get_SSIM(prediction, truth, mask)
            ssim_arr = np.mean(ssim, axis=(0, 1))
            # print(ssim_arr)
            ssim_flux += ssim_arr[0]
            ssim_sst += ssim_arr[1]
            ssim_press += ssim_arr[2]

            mse_arr = (prediction - truth)**2 # b s c h w
            mse_arr = np.sum(mse_arr, axis=(0, 1, 3, 4))
            mse_flux += mse_arr[0]
            mse_sst += mse_arr[1]
            mse_press += mse_arr[2]

            amount += 1

    print(f'polynom SSIM flux = {ssim_flux / amount:.4f}')
    print(f'polynom SSIM sst = {ssim_sst / amount:.4f}')
    print(f'polynom SSIM press = {ssim_press / amount:.4f}')
