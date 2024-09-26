import numpy as np

import datetime
import os

import torch
from config import cfg
import time
import shutil
from tqdm import tqdm
# import cv2
from thop import profile
from SSIM import get_SSIM
from earlystopping import EarlyStopping
from tensorboardX import SummaryWriter


IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
if 'kth' in cfg.dataset:
    EVAL_LEN = cfg.eval_len
gpu_nums = cfg.gpu_nums
decimals = cfg.metrics_decimals



def sum_batch(data):
    return data.sum(axis=1)


def as_type(data):
    return data.astype(cfg.data_type)


class Evaluation(object):
    def __init__(self, seq_len, use_central=False, thresholds=None):
        self._total_batch_num = 0
        self._thresholds = cfg.HKO.THRESHOLDS if thresholds is None else thresholds
        self._ssim = np.zeros((seq_len,), dtype=cfg.data_type)
        self._mae = np.zeros((seq_len,), dtype=cfg.data_type)
        self._mse = np.zeros((seq_len,), dtype=cfg.data_type)
        self._seq_len = seq_len
        self._use_central = use_central

    def clear_all(self):
        self._total_batch_num = 0
        self._ssim[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0

    def update(self, gt, pred):
        batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape

        if self._use_central:
            # Crop the central regions for evaluation
            central_region = cfg.HKO.CENTRAL_REGION
            pred = pred[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]
            gt = gt[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]

        self._total_batch_num += batch_size

        # TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        ssim = get_SSIM(prediction=pred, truth=gt)
        # bw = cfg.HKO.BALANCING_WEIGHTS
        # weights = get_balancing_weights_numba(data=gt, base_balancing_weights=bw, thresholds=self._thresholds)

        # # S*B*1*H*W
        # balanced_mse = (weights * np.square(pred - gt)).sum(axis=(2, 3, 4))
        # balanced_mae = (weights * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        mse = np.square(pred - gt).sum(axis=(2, 3, 4))
        mae = np.abs(pred - gt).sum(axis=(2, 3, 4))
        # hits, misses, false_alarms, correct_negatives = get_hit_miss_counts_numba(prediction=pred, truth=gt,
        #                                                                           thresholds=self._thresholds)

        self._ssim += sum_batch(ssim)
        # self._balanced_mse += sum_batch(balanced_mse)
        # self._balanced_mae += sum_batch(balanced_mae)
        self._mse += sum_batch(mse)
        self._mae += sum_batch(mae)
        # self._total_hits += sum_batch(hits)
        # self._total_misses += sum_batch(misses)
        # self._total_false_alarms += sum_batch(false_alarms)
        # self._total_correct_negatives += sum_batch(correct_negatives)

    def get_metrics(self):
        ssim = self._ssim / self._total_batch_num
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        l_all = [ssim, mse, mae]
        return l_all


def normalize_data_cuda(batch):
    #print(type(batch))
    batch = torch.permute(batch, (1, 0, 2, 3, 4))  # S x B x C x H x W
    batch = batch / 255.0
    return batch.cuda()


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= gpu_nums
    return rt


# is main process ?
def is_master_proc(gpu_nums=gpu_nums):
    return torch.distributed.get_rank() % gpu_nums == 0


def nan_to_num(metrics):
    for i in range(len(metrics)):
        metrics[i] = np.nan_to_num(metrics[i])
    return metrics
