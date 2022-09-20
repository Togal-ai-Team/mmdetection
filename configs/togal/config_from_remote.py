optimizer = dict(
    type='SGD',
    lr=0.0002,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='poly', power=0.99998, by_epoch=False, min_lr=3e-05)
checkpoint_config = dict(interval=10000)
log_config = dict(
    interval=100,
    hooks=[
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection',
                name='yolox_s_default_poly_lr_schedule_grayscale_augs')),
        dict(type='TextLoggerHook')
    ])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=100000, priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=100000,
        interval=10000,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/togal_yolox_s_8x8_300e_coco/latest.pth'
resume_from = None
workflow = [('train', 10000), ('val', 1)]
wandb_project = 'mmdetection'
wandb_experiment_name = 'yolox_s_default_poly_lr_schedule_grayscale_augs'
img_scale = (640, 640)
batch_size = 8
init_cfg = dict(
    type='Kaiming',
    layer='Conv2d',
    a=2.23606797749979,
    distribution='uniform',
    mode='fan_in',
    nonlinearity='leaky_relu')
model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=2.23606797749979,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=2.23606797749979,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=9,
        in_channels=128,
        feat_channels=128,
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=2.23606797749979,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Rotate',
            'prob': 0.6,
            'level': 10,
            'img_fill_val': 255,
            'max_rotate_angle': 360
        }], [{
            'type': 'BrightnessTransform',
            'prob': 0.5,
            'level': 3
        }],
                  [{
                      'type': 'Translate',
                      'prob': 0.5,
                      'level': 2,
                      'img_fill_val': 255
                  }]]),
    dict(
        type='RandomFlip',
        direction=['horizontal', 'vertical', 'diagonal'],
        flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='CocoDataset',
    ann_file='data/annotations/train_coco.json',
    img_prefix='data/images/',
    pipeline=[
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='AutoAugment',
            policies=[[{
                'type': 'Rotate',
                'prob': 0.6,
                'level': 10,
                'img_fill_val': 255,
                'max_rotate_angle': 360
            }], [{
                'type': 'BrightnessTransform',
                'prob': 0.5,
                'level': 3
            }],
                      [{
                          'type': 'Translate',
                          'prob': 0.5,
                          'level': 2,
                          'img_fill_val': 255
                      }]]),
        dict(
            type='RandomFlip',
            direction=['horizontal', 'vertical', 'diagonal'],
            flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    filter_empty_gt=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type='CocoDataset',
        ann_file='data/annotations/train_coco.json',
        img_prefix='data/images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Rotate',
                    'prob': 0.6,
                    'level': 10,
                    'img_fill_val': 255,
                    'max_rotate_angle': 360
                }], [{
                    'type': 'BrightnessTransform',
                    'prob': 0.5,
                    'level': 3
                }],
                          [{
                              'type': 'Translate',
                              'prob': 0.5,
                              'level': 2,
                              'img_fill_val': 255
                          }]]),
            dict(
                type='RandomFlip',
                direction=['horizontal', 'vertical', 'diagonal'],
                flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        filter_empty_gt=True),
    classes=('Toilet', 'Sink', 'Shower', 'Bathtub', 'Parking Lot', 'Dryer',
             'Single Swing Door', 'Double Swing Door', 'Sliding Door'),
    val=dict(
        type='CocoDataset',
        ann_file='data/annotations/val_coco.json',
        img_prefix='data/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/annotations/val_coco.json',
        img_prefix='data/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
num_last_epochs = 100000
interval = 10000
runner = dict(type='IterBasedRunner', max_iters=100000)
evaluation = dict(save_best='auto', interval=10000, metric='bbox')
work_dir = './work_dirs/togal_yolox_s_8x8_300e_coco'
auto_resume = False
gpu_ids = range(0, 1)
