import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, export
from model import ERFNet

net = ERFNet(20, "XavierUniform", train=False)

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")
context.set_context(device_id=0)

load_checkpoint("/home/gpf/log/log0/ERFNet_1-67_248.ckpt", net=net)
net.set_train(False)
input_data = Tensor(np.zeros([1, 3, 512, 1024]).astype(np.float32))
export(net, input_data, file_name="ERFNet.msmodel", file_format="MINDIR")
