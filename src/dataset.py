from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision

import sys
sys.path.append('D:/learn/vit/vision_transformer/new/src')
from config import configs


def create_dataset_cifar10(data_home, repeat_num=1, training=True):
    """
    Create a train or eval cifar-10 dataset for vit-base

    Args:
        data_home(str): the path of dataset.
        repeat_num(int): the repeat times of dataset. Default: 1
        device_num(int): num of target devices. Default: 1
        training(bool): whether dataset is used for train or eval.

    Returns:
        dataset
    """
    if not training:
        # data_set = ds.Cifar10Dataset(data_home, num_samples=300)
        data_set = ds.Cifar10Dataset(data_home, usage="test")
    else:
        data_set = ds.Cifar10Dataset(data_home, usage="train")

    resize_height = configs.image_height
    resize_width = configs.image_width

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = vision.Rescale(1.0 / 255.0, 0.0)
    normalize_op = vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply batch operations
    if training:
        data_set = data_set.batch(batch_size=configs.batch_size, drop_remainder=True)
    else:
        data_set = data_set.batch(batch_size=1, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
