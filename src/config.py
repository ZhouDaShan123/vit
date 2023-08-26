from easydict import EasyDict as edict

configs = edict({
    'name': 'cifar10',
    'pre_trained': True,  # False
    'num_classes': 10,
    'lr_init': 0.013,  # 2P
    'batch_size': 32,
    'epoch_size': 60,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'image_height': 224,
    'image_width': 224,
    'data_path': './cifar-10-batches-bin',
    'val_data_path': './cifar-10-batches-bin',
    'device_target': 'Ascend',
    'device_id': 0,
    'keep_checkpoint_max': 2,
    'checkpoint_path': '/dataset/ViT-B_16.npz',  # Can choose .ckpt or .npz
    'onnx_filename': 'vit_base',
    'air_filename': 'vit_base',

    # optimizer and lr related
    'lr_scheduler': 'cosine_annealing',
    'lr_epochs': [30, 60, 90, 120],
    'lr_gamma': 0.3,
    'eta_min': 0.0,
    'T_max': 50,
    'warmup_epochs': 0,

    # loss related
    'is_dynamic_loss_scale': 0,
    'loss_scale': 1024,
    'label_smooth_factor': 0.1,
    'use_label_smooth': True,
})
