# files_path_prefix = '/home/aosipova/EM_ocean/'
files_path_prefix = 'E:/Nastya/Data/OceanFull/'
# files_path_prefix = 'D:/Programming/PythonProjects/Alana/Data/'
SHORT_POSTFIX = '_short'

import os
from torch.nn import Conv2d, ConvTranspose2d
import numpy as np
from collections import OrderedDict


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
cfg.model_name = 'ConvLSTM'
cfg.postfix_short = SHORT_POSTFIX
cfg.gpu = '0, 1, 2, 3'
cfg.gpu_nums = len(cfg.gpu.split(','))
cfg.work_path = 'MS-RNN'
cfg.dataset = 'DWD-12-480'  # moving-mnist-20  kth_160_png  taxiBJ  HKO-7-180-with-mask  MeteoNet-120  DWD-12-480  RAIN-F
if ('HKO' in cfg.dataset) or ('MeteoNet' in cfg.dataset) or ('DWD' in cfg.dataset) or ('RAIN-F' in cfg.dataset):
    cfg.data_path = 'Precipitation-Nowcasting'
else:
    cfg.data_path = 'Spatiotemporal'
cfg.lstm_hidden_state = 64
cfg.kernel_size = 3
cfg.batch = int(4 / len(cfg.gpu.split(',')))
cfg.LSTM_conv = Conv2d
cfg.LSTM_deconv = ConvTranspose2d
cfg.CONV_conv = Conv2d

cfg.width = 91
cfg.height = 81
cfg.in_len = 7
cfg.out_len = 5
cfg.epoch = 1

cfg.early_stopping = False
cfg.early_stopping_patience = 3
if 'mnist' in cfg.dataset:
    cfg.valid_num = int(cfg.epoch * 0.5)
else:
    cfg.valid_num = int(cfg.epoch * 1)
cfg.valid_epoch = cfg.epoch // cfg.valid_num
cfg.LR = 0.0003
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
cfg.GLOBAL.DATASET_PATH = os.path.join(cfg.root_path, cfg.data_path, 'dataset', cfg.dataset)

cfg.HKO = OrderedEasyDict()
cfg.HKO.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.HKO.CENTRAL_REGION = (120, 120, 360, 360)
cfg.HKO.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)