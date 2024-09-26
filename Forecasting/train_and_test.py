
from config import cfg
import numpy as np
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import shutil
import pandas as pd
import time
from thop import profile

def train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, loader, train_sampler):
    train_valid_metrics_save_path, model_save_path, writer, save_path, test_metrics_save_path = [None] * 5
    train_loader, test_loader, valid_loader = loader
    start = time.time()
    if 'kth' in cfg.dataset:
        eval_ = Evaluation(seq_len=IN_LEN + EVAL_LEN - 1, use_central=False)
    else:
        eval_ = Evaluation(seq_len=IN_LEN + OUT_LEN - 1, use_central=False)
    if is_master_proc():
        save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        model_save_path = os.path.join(save_path, 'models')
        os.makedirs(model_save_path)
        log_save_path = os.path.join(save_path, 'logs')
        os.makedirs(log_save_path)
        # test_metrics_save_path = os.path.join(save_path, "test_metrics.xlsx")
        writer = SummaryWriter(log_save_path)
    train_loss = 0.0
    params_lis = []
    eta = 1.0
    delta = 1 / (train_epoch * len(train_loader))
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True)
    for epoch in range(1, train_epoch + 1):
        if is_master_proc():
            print('epoch: ', epoch)
        pbar = tqdm(total=len(train_loader), desc="train_batch", disable=not is_master_proc())
        # train
        train_sampler.set_epoch(epoch)
        model.train()
        for idx, train_batch in enumerate(train_loader, 1):
            #print(type(train_batch))
            train_batch = normalize_data_cuda(train_batch)
            optimizer.zero_grad()
            train_pred, decouple_loss = model([train_batch, eta, epoch], mode='train')
            loss = criterion(train_batch[1:, ...], train_pred, decouple_loss)
            loss.backward()
            optimizer.step()
            loss = reduce_tensor(loss)  # all reduce
            train_loss += loss.item()
            eta -= delta
            eta = max(eta, 0)
            pbar.update(1)

            # compute Params and FLOPs for Generator
            if epoch == 1 and idx == 1 and is_master_proc():
                Total_params = 0
                Trainable_params = 0
                NonTrainable_params = 0
                for param in model.parameters():
                    mulValue = param.numel()
                    Total_params += mulValue
                    if param.requires_grad:
                        Trainable_params += mulValue
                    else:
                        NonTrainable_params += mulValue
                Total_params = np.around(Total_params / 1e+6, decimals=decimals)
                Trainable_params = np.around(Trainable_params / 1e+6, decimals=decimals)
                NonTrainable_params = np.around(NonTrainable_params / 1e+6, decimals=decimals)
                flops, _ = profile(model.module, inputs=([train_batch, eta, epoch], 'train',))
                flops = np.around(flops / 1e+9, decimals=decimals)
                params_lis.append(Total_params)
                params_lis.append(Trainable_params)
                params_lis.append(NonTrainable_params)
                params_lis.append(flops)
                print(f'Total params: {Total_params}M')
                print(f'Trained params: {Trainable_params}M')
                print(f'Untrained params: {NonTrainable_params}M')
                print(f'FLOPs: {flops}G')
        pbar.close()

        # valid
        if epoch % valid_epoch == 0:
            train_loss = train_loss / (len(train_loader) * valid_epoch)
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for valid_batch in valid_loader:
                    valid_batch = normalize_data_cuda(valid_batch)
                    valid_pred, decouple_loss = model([valid_batch, 0, train_epoch], mode='test')
                    loss = criterion(valid_batch[1:, ...], valid_pred, decouple_loss)
                    loss = reduce_tensor(loss)  # all reduce
                    valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_loader)
            if is_master_proc():
                writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)  # plot loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
            train_loss = 0.0
            # early stopping
            if cfg.early_stopping:
                early_stopping(valid_loss, model, is_master_proc())
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break

    # test
    eval_.clear_all()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = normalize_data_cuda(test_batch)
            test_pred, decouple_loss = model([test_batch, 0, train_epoch], mode='test')
            loss = criterion(test_batch[1:, ...], test_pred, decouple_loss)
            test_loss += loss.item()
            test_batch_numpy = test_batch.cpu().numpy()
            test_pred_numpy = np.clip(test_pred.cpu().numpy(), 0.0, 1.0)
            eval_.update(test_batch_numpy[1:, ...], test_pred_numpy)

    if is_master_proc():
        test_metrics_lis = eval_.get_metrics()
        test_loss = test_loss / len(test_loader)
        test_metrics_lis.append(test_loss)
        end = time.time()
        running_time = np.around((end - start) / 3600, decimals=decimals)
        print("===============================")
        print('Running time: {} hours'.format(running_time))
        print("===============================")
        print(f'Test SSIM: {test_loss}')
        eval_.clear_all()

    if is_master_proc():
        writer.close()
