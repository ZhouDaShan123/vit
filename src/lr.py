import math
import numpy as np


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """warmup lr at the end of training"""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr1 = float(init_lr) + lr_inc * current_step
    return lr1


def warmup_cosine_annealing_lr(lr5, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """warmup cosine annealing lr"""
    base_lr = lr5
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr5 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr5)

    return np.array(lr_each_step).astype(np.float32)


def lr_steps_imagenet(_cfg, steps_per_epoch):
    """init lr step"""
    if _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         _cfg.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr