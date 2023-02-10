# 目标精度: 70.7%(来自pytorch实测)
# 这里的实现达到了70.9%

# ERFNet

使用mindpsore复现ERFNet[[论文]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf).这个项目完全迁移于原作者对ERFNet的Pytorch实现[[HERE](https://github.com/Eromera/erfnet_pytorch)].

## 关于目标精度

| (Val IOU/Test IOU) | [erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch) | [论文](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf) |
|-|-|-|
| **512 x 1024** |  **72.1/69.8** | * |
| **1024 x 2048** | * | **70.0/68.0** |

[erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch)是作者对erfnet的pytroch实现, 
上表显示了其readme中称能达到的结果和论文中声称的结果.

测试和训练时图片的输入大小尺寸会影响精度, cityscapes数据集中的图片尺寸全部是2048x1024. 论文和pytorch的具体实现, 对于图片尺寸的处理也有所不同. 
+ 论文中声称对图片和标签进行2倍下采样(1024x512)再进行训练, 测试时在1024x512下进行推断, 然后对prediction进行插值到2048x1024再和label计算IOU.
+ 在pytorch的实现中, 训练和测试均在下采样后的1024x512下进行. 

实测Pytorch实现在val上能达到70.7%的IOU.使用mindspore对erfnet进行复现, 多卡训练精度达到70.9%的IOU.

## 环境

Ascend

## 数据集

[**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): 要求数据解压完毕, 在本地的目录如下所示

```
├── gtFine .................................. ground truth
└── leftImg8bit ............................. 训练集&测试集&验证集
```

修改config.py文件中, **CityScapesRoot**变量
```py
CityScapesRoot = "/path/to/cityscapes" # 数据集根目录
```

## 训练

训练分为四步, 训练Encoder分为两步, 然后训练整个模型又分为两步.

| 阶段 | 模型 | epoch | dropout | random crop策略 |
|-|-|-|-|-|
| 1 | Encoder | 1-65 | 0.3 | 1.2固定比例随机裁剪 |
| 2 | Encoder | 66-150 | 0.3 | 1.3随机比例随机裁剪 |
| 3 | ERFNet | 1-65 | 0.2 | 1.3随机比例随机裁剪 |
| 4 | ERFNet | 66-150 | 0.3 | 1.3随机比例随机裁剪 |

在run_8p.sh中, 修改**RANK_TABLE_FILE**.

在run_8p.sh脚本中, 
+ line8: 设置训练的模型, **encode="true"** 表示训练encoder, **encode="false"** 表示训练ERFNet.
+ line36: 设置预训练的ckpt的绝对路径, **--pretrain ''** 表示从头训练.
+ line37: 设置开始的epoch数, **--start_epoch 0**表示从头训练.
+ line38: 设置是否在Encoder前加上decoder, **--attach_decoder false**, 在第三阶段从训练encoder转为训练ERFnet时,设置为true, 其余时间false.
+ line39: 设置早停epoch, **--early_stop_epoch 65**表示在第65个epoch停止训练

在model.py脚本中, 
+ line77: 1,2,4阶段使用这行, 注释掉78行.
+ line78: 在3阶段使用这行, 注释掉77行.

在dataset.py脚本中,
+ line58: 2, 3, 4阶段使用这行, 注释掉59行.
+ line59: 1阶段使用这行, 注释掉58行
+ line71: 2, 3, 4阶段使用这行, 注释掉72行.
+ line72: 1阶段使用这行, 注释掉71行.

### 1. Encoder前65个epoch

进入run_8p.sh, 修改第八行为
```sh
encode="true"
```
36-38行为
```sh
--pretrain '' \
--start_epoch 0 \
--attach_decoder false \
--early_stop_epoch 65 \
```
键入
```sh
nohup bash run_8p.sh &
```
训练完毕后, ckpt文件在**log/log0**.

### 2. Encoder的66-150epoch

记录第65个epoch的ckpt文件(Encoder-65_*.ckpt), 进入run_8p.sh修改36-39行为
```sh
--pretrain '/path/to/ERFNet/log/log0/Encoder-65_*.ckpt' \
--start_epoch 65 \
--attach_decoder false \
--early_stop_epoch 9999 \
```

进入dataset.py文件
修改random crop策略.

键入
```sh
nohup bash run_8p.sh &
```
### 3. ERFNet前65个epoch

记录第150个epoch的ckpt文件(Encoder_1-85_*.ckpt), 进入run_8p.sh
修改第八行
```sh
encode="false"
```
修改36-38行为
```sh
--pretrain '/path/to/ERFNet/log/log0/Encoder_1-85_*.ckpt' \
--start_epoch 0 \
--attach_decoder true \
--early_stop_epoch 65 \
```

进入model.py, 修改dropout参数.

键入
```sh
nohup bash run_8p.sh &
```

### 4. ERFNet的66-150epoch

记录第65个epoch的ckpt文件(ERFNet-65_*.ckpt), 进入run_8p.sh
修改第八行
```sh
encode="false"
```
修改36-38行为
```sh
--pretrain '/path/to/ERFNet/log/log0/ERFNet-65_*.ckpt' \
--start_epoch 65 \
--attach_decoder false \
--early_stop_epoch 9999 \
```

进入model.py, 修改dropout参数.

键入
```sh
nohup bash run_8p.sh &
```

很奇怪, 我就是这么调的, iou才能达标. 全阶段使用dropout0.3, random size crop最终模型效果都不够好.
并且, 改来改去确实很麻烦, 查了一下api手册, 发现根本没有这种接口, 希望mindspore给加个接口. 

## 验证

训练之后, 脚本会调用验证代码, 对不同的ckpt文件, 会加上后缀.metrics.txt, 其中包含测试精度.


## 交付

+ 模型脚本: model.py
+ 模型训练脚本: run_8p.sh, dataset.py, train_CityScapes.py, util.py, config.py
+ 模型推理脚本: infer.py, show.py
+ 能够复现目标精度和性能的模型训练超参: 在脚本中默认的参数.
+ 能够复现目标精度的模型checkpoint: ERFNet_70.9.ckpt
+ 模型readme: readme
+ 开发设计文档: 包含在readme中.
+ requirements: requirements.txt
+ 自验报告: 

## 自检
键入
```sh
python eval.py \
    --data_path /path/cityscapes \
    --run_distribute false \
    --encode false \
    --model_root_path /path/ERFNet/ERFNet_70.9.ckpt \
    --device_id 1
```
data_path为数据集根目录, model_root_path为ckpt文件路径.

验证完毕后, 会在ckpt文件同目录下生成后缀metrics.txt文件, 其中包含测试点数.

```txt
mean_iou 0.7090318296884867
mean_loss 0.296806449357143
iou_class tensor([0.9742, 0.8046, 0.9048, 0.4574, 0.5067, 0.6105, 0.6239, 0.7221, 0.9134,
        0.5903, 0.9352, 0.7633, 0.5624, 0.9231, 0.6211, 0.7897, 0.6471, 0.4148,
        0.7069], dtype=torch.float64)
```