# python eval.py --data_path /root/gpf/cityscapes/ --encode true --model_root_path /root/gpf/ERFNet_ori/log/log0/ --device_id 6

# coding=utf-8
import os
from argparse import ArgumentParser

import mindspore
import numpy as np
import random
from util import getCityLossWeight, getBool, seed_seed

from model import ERFNet
from dataset import getCityScapesDataLoader_GeneratorDataset

# mindspore lib
from mindspore import Model, Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.train.serialization import load_param_into_net, load_checkpoint, _update_param
from mindspore.context import ParallelMode
from mindspore import context

# %%
# mindspore
import mindspore
from mindspore import Tensor, context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal
from mindspore.common.parameter import Parameter
import mindspore.numpy as mnp
# 第三方库
import cv2
import numpy as np

from iouEval import iouEval
import math

from show import Colorize_cityscapes
import time

# %%
def infer(network, eval_dataloader, ckptPath, output_path):
    colorize = Colorize_cityscapes()

    # load model checkpoint
    if ckptPath is None:
        print("no model checkpoint!")
    elif not os.path.exists(ckptPath):
        print("not exist {}".format(ckptPath))
    else:
        print("load model checkpoint {}!".format(ckptPath))
        param_dict = load_checkpoint(ckptPath)
        load_param_into_net(network, param_dict)

    all_time = 0
    for index, (images, labels) in enumerate(eval_dataloader):
        begin = time.time()
        preds = network(images)
        end = time.time()
        all_time += end - begin
        preds = np.argmax(preds.asnumpy(), axis=1).astype(np.uint8)
        for i, pred in enumerate(preds):
            colorized_pred = colorize(pred)
            cv2.imwrite(os.path.join(output_path, str(index)+"_"+str(i)+".jpg"), colorized_pred)
        print("batch {} done.".format(index))

    print( "%.4f frames per sec" % (eval_dataloader.get_dataset_size()/all_time))

if __name__ == "__main__":


    # python infer.py --data_path /home/gpf/cityscapes --model_path /home/gpf/log_single_device_work/ERFNet_2-85_496.ckpt --output_path /home/gpf/output --device_id 0 > log_infer.txt

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device_id', type=int)

    config = parser.parse_args()
    model_path = config.model_path
    device_id = config.device_id
    data_path = config.data_path
    output_path = config.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    seed_seed()
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target="Ascend")
    context.set_context(device_id=device_id)
    context.set_context(save_graphs=False)

    dataloader = getCityScapesDataLoader_GeneratorDataset(data_path, "val", 32, False, 512, False, False,
                                             rank_id=0, global_size=1, repeat=1)

    weight_init = "XavierUniform"
    network = ERFNet(20, weight_init, train=False)
    network.set_train(False)

    infer(network, dataloader, model_path, output_path)