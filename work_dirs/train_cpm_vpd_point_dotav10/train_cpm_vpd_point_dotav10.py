angle_version = 'le90'
auto_resume = False
checkpoint_config = dict(interval=1)
classes = (
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter',
)
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_file='/mnt/tmp/datasets/DOTAv10/split_ss_dota/test/images/',
        classes=(
            'plane',
            'baseball-diamond',
            'bridge',
            'ground-track-field',
            'small-vehicle',
            'large-vehicle',
            'ship',
            'tennis-court',
            'basketball-court',
            'storage-tank',
            'soccer-ball-field',
            'roundabout',
            'harbor',
            'swimming-pool',
            'helicopter',
        ),
        img_prefix='/mnt/tmp/datasets/DOTAv10/split_ss_dota/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1024,
                    1024,
                ),
                transforms=[
                    dict(type='RResize'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        samples_per_gpu=4,
        type='DOTADataset',
        version='le90'),
    train=dict(
        ann_file='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles/',
        classes=(
            'plane',
            'baseball-diamond',
            'bridge',
            'ground-track-field',
            'small-vehicle',
            'large-vehicle',
            'ship',
            'tennis-court',
            'basketball-court',
            'storage-tank',
            'soccer-ball-field',
            'roundabout',
            'harbor',
            'swimming-pool',
            'helicopter',
        ),
        img_prefix='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(img_scale=(
                1024,
                1024,
            ), type='RResize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                flip_ratio=[
                    0.25,
                    0.25,
                    0.25,
                ],
                type='RRandomFlip',
                version='le90'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_bboxes',
                'gt_labels',
            ], type='Collect'),
        ],
        type='DOTADataset',
        version='le90'),
    val=dict(
        ann_file='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/annfiles/',
        classes=(
            'plane',
            'baseball-diamond',
            'bridge',
            'ground-track-field',
            'small-vehicle',
            'large-vehicle',
            'ship',
            'tennis-court',
            'basketball-court',
            'storage-tank',
            'soccer-ball-field',
            'roundabout',
            'harbor',
            'swimming-pool',
            'helicopter',
        ),
        img_prefix='/mnt/tmp/datasets/DOTAv10/split_ss_dota/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1024,
                    1024,
                ),
                transforms=[
                    dict(type='RResize'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(size_divisor=32, type='Pad'),
                    dict(type='DefaultFormatBundle'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='DOTADataset',
        version='le90'),
    workers_per_gpu=2)
data_root = '/mnt/tmp/datasets/DOTAv10/split_ss_dota/'
dataset_type = 'DOTADataset'
dist_params = dict(backend='nccl')
evaluation = dict(interval=6, metric='mAP')
find_unused_parameters = True
gpu_ids = range(0, 2)
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        4,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet',
        zero_init_residual=False),
    bbox_head=dict(
        bbox_coder=dict(angle_version='le90', type='DistanceAnglePointCoder'),
        center_sample_radius=1.5,
        center_sampling=True,
        centerness_on_reg=True,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='RotatedIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_on_bbox=True,
        num_classes=15,
        regress_ranges=(
            (
                -1,
                32,
            ),
            (
                32,
                64,
            ),
            (
                64,
                128,
            ),
            (
                128,
                256,
            ),
            (
                256,
                512,
            ),
            (
                512,
                100000000.0,
            ),
        ),
        scale_angle=True,
        separate_angle=False,
        stacked_convs=4,
        strides=[
            4,
            8,
            16,
            32,
            64,
            128,
        ],
        type='CPMVPDHead'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=6,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_thr=0.1),
        nms_pre=2000,
        score_thr=0.05,
        store_dir='/mnt/tmp/PointOBB-v2/exps/exp1/cpm_vpd_point_dotav10/'),
    train_cfg=dict(
        alpha=1,
        cls_weight=20,
        js_weight=1.0,
        store_dir='/mnt/tmp/PointOBB-v2/exps/exp1/cpm_vpd_point_dotav10/',
        thresh1=8,
        use_point_supervised=True,
        visualize=True),
    type='RotatedFCOS')
mp_start_method = 'fork'
opencv_num_threads = 0
optimizer = dict(lr=0.05, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
resume_from = None
runner = dict(max_epochs=6, type='EpochBasedRunner')
store_dir = '/mnt/tmp/PointOBB-v2/exps/exp1/cpm_vpd_point_dotav10/'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            1024,
            1024,
        ),
        transforms=[
            dict(type='RResize'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(size_divisor=32, type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(img_scale=(
        1024,
        1024,
    ), type='RResize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        flip_ratio=[
            0.25,
            0.25,
            0.25,
        ],
        type='RRandomFlip',
        version='le90'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(size_divisor=32, type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
    ], type='Collect'),
]
work_dir = './work_dirs/train_cpm_vpd_point_dotav10'
workflow = [
    (
        'train',
        1,
    ),
]
