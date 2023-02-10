# coding=utf-8
# %%
from util import getCityLossWeight
from config import TrainConfig, EvalConfig, CityScapesRoot, ms_train_data, attach_decoder
from config import weight_init, train_encode, run_distribute
from iouEval import iouEval
from model import ERFNet, Encoder_pred
from dataset import getCityScapesDataLoader_mindrecordDataset

# mindspore lib
from mindspore import Model, Tensor
from mindspore.train.loss_scale_manager import FixedLossScaleManager, DynamicLossScaleManager
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, \
    TimeMonitor, LossMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint, _update_param
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import get_rank, get_group_size, init

# std lib
import os

# 3rdparty lib
import numpy as np
import torch

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
            factor = self.pow(1-self.exp(-loss), self.gamma) * self.alpha
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

class LossMonitor_mine(Callback):
    def __init__(self, per_print_times, learning_rate):
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.loss_list = []
        self.learning_rate = learning_rate

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print("epoch:%d lr: %s" % (cb_params.cur_epoch_num, self.learning_rate[cb_params.cur_step_num]))

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self.loss_list.append(loss)
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss))
            print("average loss is %s" % (np.mean(self.loss_list)))
            print()

    def epoch_end(self, run_context):
        self.loss_list = []


def copy_param(erfnet, encoder_pretrain_path):
    print("attach decoder.")
    encoder_trained_par = load_checkpoint(encoder_pretrain_path)
    erfnet_par = erfnet.parameters_dict()
    for name, param_old in encoder_trained_par.items():
        if name.startswith("encoder"):
            _update_param(erfnet_par[name], encoder_trained_par[name])

# %%
def train(trainConfig):
    rank_id = 0
    rank_size = 1
    if run_distribute:
        context.set_auto_parallel_context(parameter_broadcast=True)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()
        rank_id = get_rank()
        rank_size = get_group_size()

    encode = train_encode
    save_prefix = "Encoder" if train_encode else "ERFNet"
    if trainConfig.epoch == 0:
        raise RuntimeError("?")

    if encode:
        network = Encoder_pred(trainConfig.num_class, weight_init)
    else:
        network = ERFNet(trainConfig.num_class, weight_init)
    network.set_train(True)
    
    dataloader = getCityScapesDataLoader_mindrecordDataset(ms_train_data, 6, \
        encode, trainConfig.train_img_size, shuffle=True, aug=True, rank_id=rank_id, global_size=rank_size, repeat=trainConfig.repeat)

    if trainConfig.if_adam:
        # opt = nn.Adam(network.trainable_params(), trainConfig.learning_rate, weight_decay=1e-4, eps=1e-08, loss_scale=32)
        opt = nn.Adam(network.trainable_params(), trainConfig.learning_rate, weight_decay=1e-4, eps=1e-08)
    else:
        opt = nn.Momentum(network.trainable_params(), trainConfig.learning_rate, \
            momentum=trainConfig.momentum, weight_decay=trainConfig.weight_decay)

    loss = SoftmaxCrossEntropyLoss(trainConfig.num_class, getCityLossWeight(encode))

    # load model checkpoint
    if trainConfig.model_load_path_for_train is None or not os.path.exists(trainConfig.model_load_path_for_train):
        print("no model checkpoint! {}".format(trainConfig.model_load_path_for_train))
    else:
        print("load model checkpoint {}!".format(trainConfig.model_load_path_for_train))

        # 训练完encoder后, 嵌入ERFnet内
        if attach_decoder:
            copy_param(network, trainConfig.model_load_path_for_train)
        else:
            param_dict = load_checkpoint(trainConfig.model_load_path_for_train)
            load_param_into_net(network, param_dict)

    # manager_loss_scale = FixedLossScaleManager(32, drop_overflow_update=True)
    loss_scale_manager = DynamicLossScaleManager()
    wrapper = Model(network, loss, opt, loss_scale_manager=loss_scale_manager, keep_batchnorm_fp32=True)
    # wrapper = Model(network, loss, opt)

    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps= \
                                    trainConfig.epoch_num_save * dataloader.get_dataset_size(), \
                                    keep_checkpoint_max=9999)
        saveModel_cb = ModelCheckpoint(prefix=save_prefix, directory= \
            trainConfig.model_save_path_for_train, config=config_ck)
        call_backs = [saveModel_cb, LossMonitor_mine(1, trainConfig.learning_rate.asnumpy())]
    else:
        call_backs = [LossMonitor_mine(1, trainConfig.learning_rate.asnumpy())]

    print("============== Starting {} Training ==============".format(save_prefix))
    wrapper.train(trainConfig.epoch, dataloader, callbacks=call_backs, dataset_sink_mode=False)
    return network

if __name__ == "__main__":
    train(TrainConfig())
