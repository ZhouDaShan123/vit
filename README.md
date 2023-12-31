# 数据集

CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图 片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。

CIFAR-10 是 3 通道的彩色 RGB 图像，图片的尺寸为 32×32 ，数据集中一共有 50000 张训练图片和 10000 张测试图片。5个训练批次 + 1个测试批次，每一批 10000 张图片。测试批次包含 10000 张图片，是由每一类图片随机抽取出 1000 张组成的集合。训练批次是由剩下的 50000 张图片打乱顺序，然后随机分成5份，所以可能某个训练批次中10个种类的图片数量不是对等的，会出现一个类的图片数量比另一类多的情况。

数据集官方提供有三个版本python，matlab 和 binary version，本文选择的是 binary version。

下载地址（三个版本）：

http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz

http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

数据集解压之后的目录结构：

 ```text
└─cifar-10-batches-bin
    ├─batches.meta.txt
    ├─cdata_batch_1.bin
    ├─cdata_batch_2.bin
    ├─cdata_batch_3.bin
    ├─cdata_batch_4.bin
    ├─cdata_batch_5.bin
    ├─readme.html
    └─test_batch.bin
```

# 环境要求

- 硬件
    - 硬件后端可使用 Ascend or GPU or CPU
- 框架
    - MindSpore2.0


# 模型训练和推理

MindSpore安装和学习的资源如下：
   - [MindSpore2.0安装教程](https://www.mindspore.cn/install)
   - [MindSpore初学入门](https://www.mindspore.cn/tutorials/zh-CN/r2.0/index.html)
   - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/r2.0/index.html)


通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估。

模型训练可以从 0 开始训练，这样训练的时间会久一些，也可以下载预训练权重文件，这个预训练权重文件是 google 官方基于 [ImageNet21k](https://console.cloud.google.com/storage/vit_models/) 的预训练模型 [ViT-B_16](http://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) 。注意这里是 .npz 格式的文件（仅适用于ViT-base-16），需要将 `config.py` 文件的 `checkpoint_path` 参数改为 .npz 文件的路径。也可手动将 .npz 格式的权重文件转换为 MindSpore 支持的 ckpt 格式的权重文件，具体见代码中的 npz_converter.py 文件。

代码仓链接：

代码结构如下：

 ```text
└─Vit
    ├─src
        ├─config.py
        ├─cifar-10-batches-bin
        ├─cifar-10-batches-bin
        ├─cifar-10-batches-bin
        └─cifar-10-verify-bin
    ├─train.py
    └─eval.py
```



- Ascend or GPU 处理器环境运行

  ```python
  # 运行训练示例

  python train.py
  
  # 运行推理示例

  python eval.py

# 结果展示

训练结果部分展示如下：

 ```python
    nohup: ignoring input
    batch_num: 1875
    Model load over!
    Momentum cfg.weight_decay:  0.0001
    args_opt.do_val:  False
    train begin!
    cfg.epoch_size: 200
    epoch: 1 step: 10, loss is 0.1347767412662506
    epoch: 1 step: 20, loss is 0.2614998519420624
    epoch: 1 step: 30, loss is 0.3215785026550293
    epoch: 1 step: 40, loss is 0.3495117425918579
    ...
    epoch: 11 step: 1860, loss is 0.09093411266803741
    epoch: 11 step: 1870, loss is 0.24195249378681183
    Train epoch time: 955440.238 ms, per step time: 509.568 ms
    epoch: 12 step: 5, loss is 0.11231812089681625
    epoch: 12 step: 15, loss is 0.1329130381345749
    epoch: 12 step: 25, loss is 0.12713535130023956
    epoch: 12 step: 35, loss is 0.1593036949634552
    epoch: 12 step: 45, loss is 0.14036336541175842
    ...
    epoch: 60 step: 1825, loss is 0.00012944571790285408
    epoch: 60 step: 1835, loss is 0.0003866434271913022
    epoch: 60 step: 1845, loss is 0.00015355084906332195
    epoch: 60 step: 1855, loss is 0.0009922076715156436
    epoch: 60 step: 1865, loss is 0.0023577185347676277
    epoch: 60 step: 1875, loss is 6.623654189752415e-05
    Train epoch time: 952240.974 ms, per step time: 507.862 ms
    Train success！
    
```

推理结果展示如下：

```python
    nohup: ignoring input
    batch_num: 10000
    Param load success!
    model eval begin!
    model's accuracy is {'acc': 0.9902}
```
