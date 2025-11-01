import datetime
import os
import shutil
import time


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from Forecasting.utils import *


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
            train_batch = normalize_data(train_batch, cfg.min_vals, cfg.max_vals)
            input = train_batch[:, :cfg.in_len, :cfg.channels].clone()
            train_batch = train_batch.cuda()
            input = input.cuda()
            # train_pred = model(input)
            mu_seq, sigma2_seq = model(input)
            # print(mu_seq.shape)
            loss = criterion(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels],
                             mu_seq[:, :cfg.out_len], sigma2_seq[:, :cfg.out_len])
            # loss = criterion(train_batch[:, cfg.in_len:cfg.in_len + cfg.out_len, :cfg.channels],
            #                  train_pred[:, :, :cfg.channels], mask)

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
        if cfg.DELETE_OLD_MODEL and os.path.exists(model_save_path):
            os.remove(model_save_path)
        # model_save_path = os.path.join(save_path, 'models')
        log_save_path = os.path.join(save_path, 'logs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(model_save_path)
            os.makedirs(log_save_path)
        # logs_file.write(f'Saving model to {model_save_path}')
        print(f'Saving model to {model_save_path}')
        torch.save(model.state_dict(), f'{model_save_path}')
    return