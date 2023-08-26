"""model eval"""
import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore import set_seed
from mindspore.train.model import Model
from mindspore import load_checkpoint, load_param_into_net

from src.config import configs
from src.dataset import create_dataset_cifar10
from src.models import VisionTransformer
import src.net_config as configs_net

set_seed(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='vit_base')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10'],
                        help='dataset name.')
    parser.add_argument('--sub_type', type=str, default='ViT-B_16',
                        choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'testing'])
    parser.add_argument('--checkpoint_path', type=str,
                        default='./',
                        help='checkpoint file path')
    parser.add_argument('--device_target', type=str, default='GPU', help='device target Ascend or GPU. (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend or GPU. (Default: 0)')
    args_opt = parser.parse_args()

    cfg = configs

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    dataset = create_dataset_cifar10(
        data_home=cfg.val_data_path,
        repeat_num=1,
        training=False)
    batch_num = dataset.get_dataset_size()
    print('batch_num:', batch_num)

    CONFIGS = {'ViT-B_16': configs_net.get_b16_config,
               'ViT-B_32': configs_net.get_b32_config,
               'ViT-L_16': configs_net.get_l16_config,
               'ViT-L_32': configs_net.get_l32_config,
               'ViT-H_14': configs_net.get_h14_config}

    net = VisionTransformer(CONFIGS[args_opt.sub_type], num_classes=cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), 0.01, cfg.momentum, weight_decay=cfg.weight_decay)

    param_dict = load_checkpoint(args_opt.checkpoint_path)
    print("Checkpoint: {}".format(args_opt.checkpoint_path))
    load_param_into_net(net, param_dict)
    print("Param load success!")

    net.set_train(False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    print("model eval begin!")
    acc = model.eval(dataset)
    print(f"model's accuracy is {acc}")
