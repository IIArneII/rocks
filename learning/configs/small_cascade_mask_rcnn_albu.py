
_base_ = [
    'mmdet::_base_/default_runtime.py',
    'mmdet::_base_/models/cascade-mask-rcnn_r50_fpn.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/datasets/coco_instance.py',
]

device = 'cuda'

model = dict(
    backbone=dict(
        depth=50,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
    roi_head=dict(
        mask_head=dict(num_classes=1),
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1)
        ]),
)

albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='RandomGamma', p=0.3, gamma_limit=(80, 120)),
    dict(type='CLAHE',       p=0.3, clip_limit=(1, 4), tile_grid_size=(8, 8)),
    dict(type='RandomFog',   p=0.2, fog_coef_lower=0.1,
                             fog_coef_upper=0.3, alpha_coef=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.1,
            ),
        keymap=dict(img='image', gt_bboxes='bboxes',
                    gt_masks='masks'),
        skip_img_without_anno=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

data_root = 'D:/Projects/rock_frag/sam2/data/'
metainfo = dict(classes=('rock', 'ladle', 'ladle_handle',))

train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco/train.json',
        data_prefix=dict(img='raw_frames/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco/val.json',
        data_prefix=dict(img='raw_frames/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.0)}))
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=500
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends)

val_evaluator = dict(
    ann_file=f'{data_root}coco/val.json',
    metric=['bbox', 'segm']
)
test_evaluator = dict(
    ann_file=f'{data_root}coco/val.json',
    metric=['bbox', 'segm']
)

