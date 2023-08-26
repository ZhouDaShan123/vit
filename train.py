import argparse

from mindspore import context, Tensor, nn
from mindspore.train import Model
from mindspore.train import CheckpointConfig, LossMonitor, ModelCheckpoint, TimeMonitor
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net

from src.models import VisionTransformer
from src.dataset import create_dataset_cifar10
from src.config import configs
from src.lr import lr_steps_imagenet
from src.eval_callback import EvalCallBack
import src.net_config as configs_net
from src.npz_converter import npz2ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10'],
                        help='dataset name.')
    parser.add_argument('--data_home', type=str, default='./cifar-10-batches-bin',
                        help='dataset name.')
    parser.add_argument('--sub_type', type=str, default='ViT-B_16',
                        choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14'])
    parser.add_argument('--pre_train', type=bool, default=True, help='use Pre-training or not.')
    parser.add_argument('--pre_checkpoint', type=str,
                        default='./ViT-B_16.npz',
                        help='checkpoint file path')
    parser.add_argument('--device_target', type=str, default='GPU', help='CPU/GPU/Ascend.')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--device_start', type=int, default=0, help='start device id. (Default: 0)')
    parser.add_argument('--lr_init', type=float, default=None, help='start lr. (Default: None)')
    parser.add_argument('--modelarts', type=bool, default=False, help='use ModelArts or not.')
    parser.add_argument('--logs_dir', type=str, default='./ckpt/', help='dir to save logs and ckpt. (Default: ./ckpt/)')
    parser.add_argument('--do_val', type=bool, default=False, help='do evaluation. (Default: False)')
    args_opt = parser.parse_args()

    cfg = configs

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    dataset = create_dataset_cifar10(
        data_home=args_opt.data_home,
        repeat_num=1,
        training=True)
    batch_num = dataset.get_dataset_size()
    print('batch_num:', batch_num)

    CONFIGS = {'ViT-B_16': configs_net.get_b16_config,
               'ViT-B_32': configs_net.get_b32_config,
               'ViT-L_16': configs_net.get_l16_config,
               'ViT-L_32': configs_net.get_l32_config,
               'ViT-H_14': configs_net.get_h14_config}

    net = VisionTransformer(CONFIGS[args_opt.sub_type], num_classes=cfg.num_classes)

    # 需不需要加载预训练权重
    print('Pre-training :', args_opt.pre_train)
    print("Pretrain checkpoint: {}".format(args_opt.pre_checkpoint))

    if args_opt.pre_train:
        if args_opt.pre_checkpoint.endswith(".ckpt"):
            param_dict = load_checkpoint(args_opt.pre_checkpoint)
        elif args_opt.pre_checkpoint.endswith(".npz"):
            if args_opt.sub_type == 'ViT-B_16':
                param_dict = npz2ckpt(args_opt.pre_checkpoint)
            else:
                raise NotImplementedError('Unsupported model type for convert.')
        else:
            raise ValueError("Unsupported checkpoint format.")

        load_param_into_net(net, param_dict)
        print("Param load success!")

    lr = lr_steps_imagenet(cfg, batch_num)

    print('Momentum cfg.weight_decay: ', cfg.weight_decay)
    opt = nn.Momentum(
        params=net.trainable_params(),
        learning_rate=Tensor(lr),
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    model = Model(net,
        loss_fn=loss,
        optimizer=opt,
        metrics={'acc'},
        keep_batchnorm_fp32=False)

    # 训练过程参数配置
    config_ck = CheckpointConfig(
        save_checkpoint_steps=batch_num,
        keep_checkpoint_max=cfg.keep_checkpoint_max,
    )

    time_cb = TimeMonitor(data_size=1)
    ckpoint_cb = ModelCheckpoint(
        prefix="train_vit_",
        directory=args_opt.logs_dir,
        config=config_ck,
    )
    loss_cb = LossMonitor(per_print_times=1)

    # 训练过程推理配置
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(
        model,
        create_dataset_cifar10(data_home=args_opt.data_home, training=False,),
        eval_per_epoch=1,
        epoch_per_eval0=epoch_per_eval,
        config=args_opt)

    args_opt.do_val = False
    print("args_opt.do_val: ", args_opt.do_val)
    if args_opt.do_val:
        cbs = [time_cb, ckpoint_cb, loss_cb, eval_cb]
    else:
        cbs = [time_cb, ckpoint_cb, loss_cb]

    print("Train begin!")
    model.train(cfg.epoch_size, dataset, callbacks=cbs, dataset_sink_mode=True)
    # model.train(cfg.epoch_size, dataset, callbacks=cbs)
    print("Train success！")




