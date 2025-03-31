# ..\venv_base\Scripts\activate
# run: torchrun --nproc_per_node=4 --master_port 39985 main.py 3 2019
import os

from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from models.encoder_decoder import Encoder_Decoder
from loss import Loss
from train_and_test import train_and_test, test
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loader import create_dataloaders, count_offset
import argparse
from collections import OrderedDict


# fix random seed
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


fix_random(2024)

gpu_nums = cfg.gpu_nums
batch_size = cfg.batch
train_epoch = cfg.epoch
valid_epoch = cfg.valid_epoch
LR = cfg.LR

parser = argparse.ArgumentParser()
# parser.add_argument("features_amount", type=int)
# parser.add_argument("start_year", type=int)
# parser.add_argument("model_name", type=int)
# parser.add_argument("local_rank", type=int, default=-1, help='node rank for distributed training')
# args = parser.parse_args()
# cfg.features_amount = args.features_amount
start_year = cfg.start_year
end_year, offset = count_offset(start_year)
# cfg.model_name = args.model_name
# torch.cuda.set_device(args.local_rank)

# parallel group
torch.distributed.init_process_group(backend="gloo")

# model
# model = Model(nets[0], nets[1], nets[2])
model = Encoder_Decoder(cfg.in_len, cfg.out_len, (cfg.batch, cfg.height, cfg.width), 3, 2, 1)


# optimizer
if cfg.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = None


model_save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH + f'/models/features_{cfg.features_amount}_epoch_{cfg.epoch}.pth'
model = model.cuda()
model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
print(f'Trying to read {model_save_path},\n exists = {os.path.exists(model_save_path)}\n')

if cfg.LOAD_MODEL and os.path.exists(model_save_path):
    print('Loading model')
    # original saved file with DataParallel
    state_dict = torch.load(model_save_path)
    # create new OrderedDict that does not contain `total`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print(k)
        if not ('total' in k):
            new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict)

threads = cfg.dataloader_thread
train_data, valid_data, test_data = create_dataloaders(cfg.root_path, start_year, end_year, cfg)

train_sampler = DistributedSampler(train_data, shuffle=True)
valid_sampler = DistributedSampler(valid_data, shuffle=False)
train_loader = DataLoader(train_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=train_sampler)
test_loader = DataLoader(test_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=False)
valid_loader = DataLoader(valid_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=valid_sampler)
loader = [train_loader, test_loader, valid_loader]

# loss
criterion = Loss().cuda()

# train valid test
if not start_year == 2019:
    train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, loader, train_sampler)

# test and plot
if start_year == 2019:
    test(model, criterion, test_loader, train_epoch, cfg, len(train_data))