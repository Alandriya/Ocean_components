import datetime
import shutil
import time

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Forecasting.utils import *
from Plotting.nn_plotter import plot_predictions
from Forecasting.SSIM import get_SSIM


def test(test_data, model, mask):
    test_sampler = DistributedSampler(test_data, shuffle=False)
    test_loader = DataLoader(test_data, num_workers=cfg.dataloader_thread, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                             sampler=test_sampler)
    days_delta1 = (datetime.datetime(2024, 1, 1, 0, 0) - datetime.datetime(1979, 1, 1, 0, 0)).days

    if is_master_proc():
        with ((torch.no_grad())):
            ssim_flux = 0.0
            mse_flux = 0.0
            time_start = time.time()
            amount = 0
            for idx, test_batch in enumerate(test_loader):
                print(f'Epoch {idx}')
                print(f'time elapsed: {int((time.time() - time_start)/60)} minutes')
                test_batch = normalize_data(test_batch, cfg.min_vals, cfg.max_vals)
                input = test_batch[:, :cfg.in_len, :cfg.channels].clone()
                # test_batch = test_batch.cuda()
                # input = input.cuda()
                test_pred_values = model(input)

                test_batch_scaled = test_batch.clone().detach()
                truth = test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len].detach().clone()
                prediction = test_pred_values.detach().clone()
                truth = truth.cpu().numpy()
                prediction = prediction.cpu().numpy()

                ssim = get_SSIM(prediction, truth[:, :, :cfg.channels], mask)
                ssim_arr = np.mean(ssim, axis=(0, 1))
                ssim_flux += ssim_arr[0]

                mse_arr = (prediction - truth[:, :, :cfg.channels]) ** 2  # b s c h w
                mse_arr = np.sum(mse_arr, axis=(0, 1, 3, 4))
                mse_flux += mse_arr[0]

                amount += 1

                for channel in range(cfg.channels):
                    test_batch_scaled[:, :cfg.in_len + cfg.out_len, channel] *= (
                                cfg.max_vals[channel] - cfg.min_vals[channel])
                    test_batch_scaled[:, :cfg.in_len + cfg.out_len, channel] += cfg.min_vals[channel]

                    test_pred_values[:, :, channel] *= (cfg.max_vals[channel] - cfg.min_vals[channel])
                    test_pred_values[:, :, channel] += cfg.min_vals[channel]

                print(cfg.min_vals)
                print(cfg.max_vals)
                print(tuple(torch.amin(test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels],
                                       dim=(0, 1, 3, 4))))
                print(tuple(torch.amax(test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels],
                                       dim=(0, 1, 3, 4))))
                print(tuple(torch.amin(test_pred_values[:, :, :cfg.channels], dim=(0, 1, 3, 4))))
                print(tuple(torch.amax(test_pred_values[:, :, :cfg.channels], dim=(0, 1, 3, 4))))

                differ = test_batch_scaled[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels] - test_pred_values
                loss_divided = torch.mean(torch.sum(differ ** 2, (3, 4)), (0, 1))

                if idx == 0:
                    for i in range(cfg.batch):
                        day = datetime.datetime(1979, 1, 1) + datetime.timedelta(days=days_delta1 + idx + i)
                        test_batch_numpy = test_batch_scaled[i, :, :cfg.channels].cpu().numpy()
                        test_pred_numpy = test_pred_values[i, :, :cfg.channels].cpu().numpy()
                        plot_predictions(cfg.root_path, test_batch_numpy[cfg.in_len:cfg.in_len + cfg.out_len],
                                         test_pred_numpy, cfg.model_name,
                                         cfg.features_amount, day, mask, cfg, postfix_simple=True)
                    break
        logs_file.write(f'SSIM flux = {ssim_flux / amount:.4f}\n MSE flux = {mse_flux / amount:.4f}')
        print(f'SSIM flux = {ssim_flux / amount:.4f}')
        print(f'MSE flux = {mse_flux / amount:.4f}')
        return