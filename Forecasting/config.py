# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'D:/Nastya/Data/OceanFull/'
# files_path_prefix = ''


import os
from collections import OrderedDict

import numpy as np


class OrderedEasyDict(OrderedDict):
    """Using OrderedDict for the `easydict` package
    See Also https://pypi.python.org/pypi/easydict/
    """

    def __init__(self, d=None, **kwargs):
        super(OrderedEasyDict, self).__init__()
        if d is None:
            d = OrderedDict()
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        # special handling of self.__root and self.__map
        if name.startswith('_') and (name.endswith('__root') or name.endswith('__map')):
            super(OrderedEasyDict, self).__setattr__(name, value)
        else:
            if isinstance(value, (list, tuple)):
                value = [self.__class__(x)
                         if isinstance(x, dict) else x for x in value]
            else:
                value = self.__class__(value) if isinstance(value, dict) else value
            super(OrderedEasyDict, self).__setattr__(name, value)
            super(OrderedEasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

cfg = OrderedEasyDict()

cfg.features_amount = 1
# ConvLSTM  MS-LSTM  Att-Unet Transformer
# cfg.model_name = 'Attention U-net'
cfg.model_name = 'SDE_HNN'
cfg.nn_mode = 'train'

cfg.bins = 100
cfg.LOAD_MODEL = False
cfg.DELETE_OLD_MODEL = True
cfg.channels = 1
cfg.A_coeff_weight = 0.3
cfg.B_coeff_weight = 0.01

cfg.gpu = '0, 1, 2, 3'
cfg.gpu_nums = len(cfg.gpu.split(','))
cfg.work_path = 'MS-RNN'
cfg.dataset = 'Ocean'
cfg.lstm_hidden_state = 32
cfg.kernel_size = 2
cfg.batch = 64

cfg.width = 91
cfg.height = 81
cfg.in_len = 14
cfg.out_len = 3
cfg.epoch = 10
flux_quantiles = np.load(files_path_prefix + f'DATA/FLUX_1979-2025_diff_quantiles.npy')
sst_quantiles = np.load(files_path_prefix + f'DATA/SST_1979-2025_diff_quantiles.npy')
press_quantiles = np.load(files_path_prefix + f'DATA/PRESS_1979-2025_diff_quantiles.npy')

cfg.min_vals = (flux_quantiles[0], sst_quantiles[0], press_quantiles[0])
cfg.max_vals = (flux_quantiles[-1], sst_quantiles[-1], press_quantiles[-1])

cfg.early_stopping = False
cfg.early_stopping_patience = 3
if 'mnist' in cfg.dataset:
    cfg.valid_num = int(cfg.epoch * 0.5)
else:
    cfg.valid_num = int(cfg.epoch * 1)
cfg.valid_epoch = cfg.epoch // cfg.valid_num
cfg.LR = 0.00035
cfg.optimizer = 'Adam'
cfg.dataloader_thread = 0
cfg.data_type = np.float32
cfg.scheduled_sampling = True
if 'PredRNN-V2' in cfg.model_name:
    cfg.reverse_scheduled_sampling = True
else:
    cfg.reverse_scheduled_sampling = False
cfg.TrajGRU_link_num = 10
cfg.ce_iters = 5
cfg.decouple_loss_weight = 0.01
cfg.la_num = 30
cfg.LSTM_layers = 6
cfg.metrics_decimals = 3

cfg.root_path = files_path_prefix

cfg.GLOBAL = OrderedEasyDict()
cfg.GLOBAL.MODEL_LOG_SAVE_PATH = os.path.join(cfg.root_path, cfg.work_path, 'save', cfg.dataset, cfg.model_name)