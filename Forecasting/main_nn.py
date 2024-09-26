# run: torchrun --nproc_per_node=4 --master_port 39985 main_nn.py 3 2019
import os
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
from loss import Loss
from train_and_test import train_and_test
from net_params import nets
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from loader import create_dataloaders, count_offset
import argparse

def cleanup():
    dist.destroy_process_group()

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


fix_random(2022)

gpu_nums = cfg.gpu_nums
batch_size = cfg.batch
train_epoch = cfg.epoch
valid_epoch = cfg.valid_epoch
LR = cfg.LR

parser = argparse.ArgumentParser()
parser.add_argument("features_amount", type=int)
parser.add_argument("start_year", type=int)
# parser.add_argument("local_rank", type=int, default=-1, help='node rank for distributed training')
args = parser.parse_args()
cfg.features_amount = args.features_amount
start_year = args.start_year
end_year, offset = count_offset(start_year)
# torch.cuda.set_device(args.local_rank)

# parallel group
torch.distributed.init_process_group(backend="gloo")

# model parallel
model = Model(nets[0], nets[1], nets[2])
model = model.cuda()
model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

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

# optimizer
if cfg.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = None

# loss
criterion = Loss().cuda()

# train valid test
train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, loader, train_sampler)