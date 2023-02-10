# coding=utf-8
import os
from argparse import ArgumentParser
from mindspore import context
from mindspore import Tensor
import mindspore
from mindspore.common.initializer import HeNormal, XavierUniform
import numpy as np
import random
import os
from util import getBool, seed_seed, getLR

parser = ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--run_distribute', type=str)
parser.add_argument('--run_online', type=str)
parser.add_argument('--encode', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--run_eval', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--repeat', type=int)
parser.add_argument('--pretrain', type=str, default="")
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--early_stop_epoch', type=int, default=9999)
parser.add_argument('--attach_decoder', type=str)

config = parser.parse_args()

run_online = getBool(config.run_online)
run_distribute = getBool(config.run_distribute)
train_encode = getBool(config.encode)
global_size = int(os.environ["RANK_SIZE"])
attach_decoder = getBool(config.attach_decoder)

context.set_context(mode = context.GRAPH_MODE)
context.set_context(device_target ="Ascend")
context.set_context(device_id=int(os.environ["DEVICE_ID"]))
context.set_context(save_graphs=False)

seed_seed() # 初始随机种子
weight_init=XavierUniform() # 权重初始化
CityScapesRoot = "/home/gpf/cityscapes" # 数据集根目录
ms_train_data = "/home/gpf/train.mindrecord" # mindrecord 训练数据
ms_val_data = "/home/gpf/val.mindrecord" # mindrecord 验证数据

# train config
class TrainConfig:
    def __init__(self):
        self.model_load_path_for_train = config.pretrain
        self.model_save_path_for_train = config.save_path
        self.subset = "train"
        self.num_class = 20

        self.if_adam = True
        if self.if_adam == False: # 使用SGD
            self.momentum = 0.9
            self.weight_decay = 1e-4

        self.epoch_num_save = 1
        self.epoch_num_eval = 1


        self.repeat = config.repeat

        self.start_epoch = config.start_epoch
        self.all_epoch = config.epoch
        
        if config.early_stop_epoch > config.epoch:
            self.epoch = config.epoch - self.start_epoch  # 不早停
        else:
            self.epoch = config.early_stop_epoch - self.start_epoch + 1 # 算入早停epoch

        self.if_train_dataset_shuffle = True

        self.train_img_size = 512
        self.learning_rate = getLR(config.lr, self.start_epoch, self.all_epoch, 496,
                                   run_distribute=run_distribute, global_size=global_size, repeat=self.repeat)

# eval config
class EvalConfig:
    def __init__(self):
        self.subset = "val"
        self.img_size = 512
