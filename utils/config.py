from easydict import EasyDict as edict
from easydict import EasyDict as edict

__C = edict()
cfg = __C


__C.manualSeed = 100
__C.CUDA = True
__C.workers = 4

__C.WBDNet = edict()
__C.WBDNet.start_epoch = 0
__C.WBDNet.batch_size = 64
__C.WBDNet.input_size = 80
__C.WBDNet.hidden_size = 20
__C.WBDNet.num_layers = 2