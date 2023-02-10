# python eval.py --data_path /root/gpf/cityscapes/ --encode true --model_root_path /root/gpf/ERFNet_ori/log/log0/ --device_id 6

# coding=utf-8
import os
from argparse import ArgumentParser

import mindspore
import numpy as np
import random
from util import getCityLossWeight, getBool, seed_seed

from iouEval import iouEval
from model import ERFNet, Encoder_pred
from dataset import getCityScapesDataLoader_GeneratorDataset
import torch

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
import numpy as np

from iouEval import iouEval
import math

# Pytorch NLLLoss + log_softmax
class SoftmaxCrossEntropyLoss(nn.Cell):

    def __init__(self, num_cls, weight):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.unsqueeze = ops.ExpandDims()
        self.get_size = ops.Size()
        self.exp = ops.Exp()
        self.pow = ops.Pow()

        self.weight = weight
        if isinstance(self.weight, tuple):
            self.use_focal = True
            self.gamma = self.weight[0]
            self.alpha = self.weight[1]
        else:
            self.use_focal = False

    def construct(self, pred, labels):
        labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1,))
        pred = self.transpose(pred, (0, 2, 3, 1))
        pred = self.reshape(pred, (-1, self.num_cls))
        one_hot_labels = self.one_hot(labels, self.num_cls, self.on_value, self.off_value)
        pred = self.cast(pred, mstype.float32)
        num = self.get_size(labels)

        if self.use_focal:
            loss = self.ce(pred, one_hot_labels)
            factor = self.pow(1 - self.exp(-loss), self.gamma) * self.alpha
            loss = self.div(self.sum(factor * loss), num)
            return loss

        if self.weight is not None:
            weight = mnp.copy(self.weight)
            weight = self.cast(weight, mstype.float32)
            weight = self.unsqueeze(weight, 0)
            expand = ops.BroadcastTo(pred.shape)
            weight = expand(weight)
            weight_masked = weight[mnp.arange(num), labels]
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss * weight_masked), self.sum(weight_masked))
        else:
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss), num)
        return loss

def IOU_1(network_trainefd, dataloader, num_class, enc):
    ioueval = iouEval(num_class)
    loss = SoftmaxCrossEntropyLoss(num_class, getCityLossWeight(enc))
    loss_list = []
    network = network_trainefd
    network.set_train(False)
    for index, (images, labels) in enumerate(dataloader):
        preds = network(images)
        l = loss(preds, labels)
        loss_list.append(float(str(l)))
        print("step {}/{}: loss:  ".format(index+1, dataloader.get_dataset_size()), l)
        preds = torch.Tensor(preds.asnumpy().argmax(axis=1).astype(np.int32)).unsqueeze(1).long()
        labels = torch.Tensor(labels.asnumpy().astype(np.int32)).unsqueeze(1).long()
        ioueval.addBatch(preds, labels)

    mean_iou, iou_class = ioueval.getIoU()
    mean_iou = mean_iou.item()
    mean_loss = sum(loss_list) / len(loss_list)
    return mean_iou, mean_loss, iou_class

# %%
def eval(network, eval_dataloader, ckptPath, encode, num_class=20, weight_init = "XavierUniform"):

    # load model checkpoint
    if ckptPath is None:
        print("no model checkpoint!")
    elif not os.path.exists(ckptPath):
        print("not exist {}".format(ckptPath))
    else:
        print("load model checkpoint {}!".format(ckptPath))
        param_dict = load_checkpoint(ckptPath)
        load_param_into_net(network, param_dict)

    mean_iou, mean_loss, iou_class = IOU_1(network, eval_dataloader, num_class, encode)
    with open(ckptPath + ".metric.txt", "w") as file:
        print("model path", ckptPath, file=file)
        print("mean_iou", mean_iou, file=file)
        print("mean_loss", mean_loss, file=file)
        print("iou_class", iou_class, file=file)

def listCKPTPath(model_root_path, enc):
    paths = []
    names = os.listdir(model_root_path)
    for name in names:
        if name.endswith(".ckpt") and name+".metric.txt" not in names:
            if enc and name.startswith("Encoder"):
                ckpt_path = os.path.join(model_root_path, name)
                paths.append(ckpt_path)
            elif not enc and name.startswith("ERFNet"):
                ckpt_path = os.path.join(model_root_path, name)
                paths.append(ckpt_path)
    return paths

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--run_distribute', type=str)
    parser.add_argument('--encode', type=str)
    parser.add_argument('--model_root_path', type=str)
    parser.add_argument('--device_id', type=int)

    config = parser.parse_args()
    model_root_path = config.model_root_path
    encode = getBool(config.encode)
    device_id = config.device_id
    CityScapesRoot = config.data_path
    run_distribute = getBool(config.run_distribute)

    seed_seed()
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target="Ascend")
    context.set_context(device_id=device_id)
    context.set_context(save_graphs=False)

    eval_dataloader = getCityScapesDataLoader_GeneratorDataset(CityScapesRoot, "val", 6, \
                                                               encode, 512, False, False)

    weight_init = "XavierUniform"
    if encode:
        network = Encoder_pred(20, weight_init, train=False)
    else:
        network = ERFNet(20, weight_init, train=False)

    if not run_distribute:
        if os.path.isdir(model_root_path):
            paths = listCKPTPath(model_root_path, encode)
            for path in paths:
                eval(network, eval_dataloader, path, encode)
        else:
            eval(network, eval_dataloader, model_root_path, encode)
    else:
        rank_id = int(os.environ["RANK_ID"])
        rank_size = int(os.environ["RANK_SIZE"])
        assert os.path.isdir(model_root_path)
        ckpt_files_path = listCKPTPath(model_root_path, encode)
        n = math.ceil(len(ckpt_files_path) / rank_size)
        ckpt_files_path = ckpt_files_path[rank_id*n : rank_id*n + n]

        for path in ckpt_files_path:
            eval(network, eval_dataloader, path, encode)
