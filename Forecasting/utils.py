import numpy as np
import os
import random
import torch
from Forecasting.config import cfg
from Forecasting.SSIM import get_SSIM
from Forecasting.loader import load_mask

IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
if 'kth' in cfg.dataset:
    EVAL_LEN = cfg.eval_len
gpu_nums = cfg.gpu_nums
decimals = cfg.metrics_decimals

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

def sum_batch(data):
    return data.sum(axis=0)


def as_type(data):
    return data.astype(cfg.data_type)


class Evaluation(object):
    def __init__(self, seq_len, use_central=False, thresholds=None, max_val=(1.0, 1.0, 1.0)):
        self._total_batch_num = 0
        self._thresholds = cfg.HKO.THRESHOLDS if thresholds is None else thresholds
        self._ssim = np.zeros((OUT_LEN, 3), dtype=cfg.data_type)
        self._mae = np.zeros((OUT_LEN,), dtype=cfg.data_type)
        self._mse = np.zeros((OUT_LEN,), dtype=cfg.data_type)
        self._seq_len = seq_len
        self._use_central = use_central
        self._max_val = max_val
        self.mask = load_mask(cfg.root_path)
        # self.csi_metrics = CriticalSuccessIndex(threshold=0.5, keep_sequence_dim=2)

    def clear_all(self):
        self._total_batch_num = 0
        self._ssim[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0
        # self.csi_metrics = CriticalSuccessIndex(threshold=0.5, keep_sequence_dim=2)

    def update(self, gt, pred):
        batch_size = gt.shape[0]
        assert gt.shape[1] == self._seq_len
        assert gt.shape == pred.shape

        if self._use_central:
            # Crop the central regions for evaluation
            central_region = cfg.HKO.CENTRAL_REGION
            pred = pred[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]
            gt = gt[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]

        self._total_batch_num += batch_size
        ssim = get_SSIM(prediction=pred, truth=gt, mask=self.mask)

        # # B*S*C*H*W
        mse = np.square(pred - gt).sum(axis=(2, 3, 4))
        mae = np.abs(pred - gt).sum(axis=(2, 3, 4))

        self._ssim += sum_batch(ssim)
        self._mse += sum_batch(mse)
        self._mae += sum_batch(mae)

    def get_metrics(self):
        ssim = self._ssim / self._total_batch_num
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        # csi = self.csi_metrics.compute()
        l_all = [ssim, mse, mae]
        return l_all


def normalize_data_cuda(batch, min_vals, max_vals):
    # print(max_vals)
    # print(min_vals)
    # print('\n\n')
    # for i in range(3, 9):
    #     print(i)
    #     print(torch.max(batch[:, :, i]))
    #     print(torch.min(batch[:, :, i]))

    for channel in range(cfg.channels):
        batch[:, :cfg.in_len + cfg.out_len, channel] = (batch[:, :cfg.in_len + cfg.out_len, channel] -
                                                        min_vals[channel]) / (max_vals[channel] - min_vals[channel])
        # normalize A
        batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels] = (batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels] -
                                                            min_vals[channel]) / (max_vals[channel] - min_vals[channel])

        # normalize B
        batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels*2] = (batch[:, :cfg.in_len + cfg.out_len, channel + cfg.channels*2] -
                                                        (min_vals[channel])) / ((max_vals[channel] - min_vals[channel]))

        #
        # # normalize eigenvalues
        # batch[:, :cfg.in_len + cfg.out_len, channel + 9 + channel*2] = (batch[:, :cfg.in_len + cfg.out_len, channel*2 + 9] -
        #                                                 min_vals[channel]) / (max_vals[channel] - min_vals[channel])
        # batch[:, :cfg.in_len + cfg.out_len, channel + 10 + channel*2] = (batch[:, :cfg.in_len + cfg.out_len, channel*2 + 10] -
        #                                                 min_vals[channel]) / (max_vals[channel] - min_vals[channel])

    # return batch.cuda()
    # print('\n\n')
    # for i in range(3, 18):
    #     print(i)
    #     print(torch.max(batch[:, :, i]))
    #     print(torch.min(batch[:, :, i]))
    # raise ValueError
    return batch


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
