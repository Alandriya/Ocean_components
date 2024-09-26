import argparse
import sys
import warnings
import os
from config import files_path_prefix, cfg
from model import Model
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
from collections import OrderedDict
from models.convlstm import ConvLSTM
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore")
from utils import *
import torch
from torch import nn
import random


def fix_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    fix_random(2024)
    np.set_printoptions(threshold=sys.maxsize)
    start_year = 2019
    end_year = 2025
    # x_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}.pt')
    # x_train = x_train[:100].clone()
    # torch.save(x_train, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_x_train_{cfg.features_amount}_short.pt')
    # del x_train
    #
    # y_train = torch.load(files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{cfg.features_amount}.pt')
    # y_train = y_train[:100].clone()
    # torch.save(y_train, files_path_prefix + f'Forecast/Train/{start_year}-{end_year}_y_train_{cfg.features_amount}_short.pt')
    # del y_train

    x_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{cfg.features_amount}.pt')
    x_test = x_test[:50].clone()
    torch.save(x_test,
               files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_x_test_{cfg.features_amount}_short.pt', )
    del x_test
    y_test = torch.load(files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{cfg.features_amount}.pt')
    y_test = y_test[:50].clone()
    torch.save(y_test, files_path_prefix + f'Forecast/Test/{start_year}-{end_year}_y_test_{cfg.features_amount}_short.pt', )
    raise ValueError
    end_year, offset = count_offset(start_year)
    mask = load_mask(files_path_prefix)
    mask_batch = np.zeros((cfg.height, cfg.width, cfg.out_len, 3))
    mask_batch[mask, :, :] = 1
    flux_array, SST_array, press_array = load_np_data(files_path_prefix, start_year, end_year)
    create_train_test(files_path_prefix, start_year, end_year, offset, cfg, flux_array, SST_array, press_array)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    hs = cfg.lstm_hidden_state
    if cfg.model_name == 'ConvLSTM':
        rnn = ConvLSTM

    # construct model
    nets = [OrderedDict({'conv_embed': [3, hs, 1, 1, 0, 1]}),
            rnn,
            OrderedDict({'conv_fc': [hs, 3, 1, 1, 0, 1]})]

    model = Model(nets[0], nets[1], nets[2])
    # model = Model(*input_args).to(device)
    loss_fn = nn.MSELoss()

    # optimizer
    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LR, momentum=0.9)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    else:
        optimizer = None

    if 'kth' in cfg.dataset:
        eval_ = Evaluation(seq_len=IN_LEN + EVAL_LEN - 1, use_central=False)
    else:
        eval_ = Evaluation(seq_len=IN_LEN + OUT_LEN - 1, use_central=False)

    train_epoch = 20
    model_save_path = files_path_prefix + f'Forecast/Checkpoints/{cfg.model_name}_{features_amount}.pth'
    def train(train_loader, train_sampler, criterion, model, valid_epoch, valid_loader, optimizer):
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
                    # 使用nn.BatchNorm2d时，flop计算会报错，需要注释掉
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
                    # writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)  # plot loss
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
                train_loss = 0.0
                # early stopping
                if cfg.early_stopping:
                    early_stopping(valid_loss, model, is_master_proc())
                    if early_stopping.early_stop:
                        print("Early Stopping!")
                        break

    def test(test_loader, model, criterion):
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


    # parallel group
    torch.distributed.init_process_group(backend="nccl")

    # model parallel
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank],
                                                output_device=args.local_rank)

    train_sampler = DistributedSampler(train_data, shuffle=True)
    valid_sampler = DistributedSampler(valid_data, shuffle=False)
    train_loader = DataLoader(train_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                              sampler=train_sampler)
    test_loader = DataLoader(test_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=False)
    valid_loader = DataLoader(valid_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                              sampler=valid_sampler)

    # Create data loaders.
    train_loader, test_loader = create_dataloaders(files_path_prefix, start_year, end_year, cfg)

    # for X, y in test_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break

    # train
    EPOCHS = 25
    checkpoint_path = files_path_prefix + f'Forecast/Checkpoints/my_checkpoint_{cfg.model_name}_{features_amount}.pth'

    if os.path.exists(checkpoint_path):
        # Loads the weights
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))


    if start_year == 2019:
        train(train_loader, train_sampler, criterion, model, valid_epoch, valid_loader, optimizer)
        test(test_loader, model, criterion)

    else:
        train(train_loader, train_sampler, criterion, model, valid_epoch, valid_loader, optimizer)