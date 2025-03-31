from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from Forecasting.utils import *
from Forecasting.config import cfg
from Plotting.nn_plotter import plot_predictions
import time
import datetime
import shutil


def train(train_data, model, criterion, optimizer, mask, model_save_path):
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(train_data, num_workers=cfg.dataloader_thread, batch_size=cfg.batch, shuffle=False, pin_memory=True,
                              sampler=train_sampler)
    train_loss = np.zeros(cfg.epoch, dtype=float)
    time_start = time.time()
    for epoch in range(1, cfg.epoch + 1):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        print(f'Epoch {epoch}/{cfg.epoch}', flush=True)
        print(f'time elapsed: {int((time.time() - time_start) / 60)} minutes')
        for idx, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            train_batch = normalize_data_cuda(train_batch, cfg.min_vals, cfg.max_vals)
            input = train_batch[:, :cfg.in_len, :cfg.channels].clone()
            train_batch = train_batch.cuda()
            input = input.cuda()
            train_pred = model(input)
            loss = criterion(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels],
                             train_pred[:, :, :cfg.channels], mask)

            loss.backward()
            optimizer.step()
            loss = reduce_tensor(loss)  # all reduce
            epoch_loss += loss.item()

        train_loss[epoch - 1] = epoch_loss
        if is_master_proc():
            print(f'Loss: {epoch_loss}', flush=True)

    if is_master_proc():
        np.save(cfg.root_path + f'Losses/loss_{cfg.model_name}.npy', train_loss)
        torch.distributed.destroy_process_group()
        save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH
        if cfg.DELETE_OLD_MODEL and os.path.exists(save_path):
            shutil.rmtree(save_path)
        # model_save_path = os.path.join(save_path, 'models')
        log_save_path = os.path.join(save_path, 'logs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(model_save_path)
            os.makedirs(log_save_path)
        print(f'Saving model to {model_save_path}')
        torch.save(model.state_dict(), f'{model_save_path}')
    return


def test(test_data, model, criterion, optimizer, mask):
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
                test_batch = normalize_data_cuda(test_batch, cfg.min_vals, cfg.max_vals)
                input = test_batch[:, :cfg.in_len, :cfg.channels].clone()
                test_batch = test_batch.cuda()
                input = input.cuda()
                test_pred_values = model(input)

                loss = criterion(test_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels],
                                 test_pred_values[:, :, :cfg.channels], mask)

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
                    # break
        print(f'SSIM flux = {ssim_flux / amount:.4f}')
        print(f'MSE flux = {mse_flux / amount:.4f}')
        return