_base_ = [
    "mmdet::convnext/cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"
]

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'

device    = "cuda"
img_size  = 600          
num_class = 1            # rock

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    roi_head=dict(
        bbox_head=[
            dict(type="Shared2FCBBoxHead", num_classes=num_class),
            dict(type="Shared2FCBBoxHead", num_classes=num_class),
            dict(type="Shared2FCBBoxHead", num_classes=num_class)
        ],
        mask_head=dict(num_classes=num_class)
    )
)

albu_train_transforms = [
    dict(type="HorizontalFlip",   p=0.5),
    dict(type="RandomBrightnessContrast", p=0.5),
    dict(type="RandomGamma",      gamma_limit=(80,120), p=0.3),
    dict(type="CLAHE",            clip_limit=(1,4),
                                  tile_grid_size=(8,8), p=0.3),
    dict(type='RandomFog',   p=0.2, fog_coef_lower=0.1,
                             fog_coef_upper=0.3, alpha_coef=0.1),
    dict(type="Sharpen",
         alpha=(0.1,0.3), lightness=(0.5,1.0), p=0.2),
    dict(type="RandomSnow", p=0.1)
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Albu",
         transforms=albu_train_transforms,
         bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels"],
            min_visibility=0.1),
         keymap=dict(img="image", gt_bboxes="bboxes",
                     gt_masks="masks"),
         skip_img_without_anno=True),
    dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
    dict(type="PackDetInputs")
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(img_size,img_size), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="PackDetInputs")
]

data_root = "D:/Projects/rock_frag/sam2/data/"
metainfo  = dict(classes=("rock",))

train_dataset = dict(
    _delete_=True,
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='CocoDataset',
        ann_file='coco/train.json',
        data_root=data_root,
        data_prefix=dict(img='raw_frames/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        filter_cfg=None,
        backend_args=None
    )
)

train_dataloader = dict(
    batch_size=3,
    num_workers=6,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco/val.json',
        data_prefix=dict(img='raw_frames/'),
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(paramwise_cfg={
    'decay_rate': 0.7,
    'decay_type': 'layer_wise',
    'num_layers': 12
})

val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=2,
)

default_hooks = dict(
    checkpoint=dict(interval=2, save_best="auto"),
    early_stop=dict(
        type="mmengine.hooks.EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        rule="greater",
        patience=10,
        priority='LOWEST'
    )
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer   = dict(type="DetLocalVisualizer",
                    vis_backends=vis_backends)

val_evaluator = dict(
    ann_file=f"{data_root}coco/val.json",
    metric=["bbox","segm"]
)
test_evaluator = dict(
    ann_file=f"{data_root}coco/val.json",
    metric=["bbox","segm"],
    format_only=False, 
    outfile_prefix='./results'
)