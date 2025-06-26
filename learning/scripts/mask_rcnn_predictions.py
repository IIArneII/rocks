from mmdet.apis import init_detector, inference_detector
from mmengine import Config
from mmengine.logging import HistoryBuffer
from torch.serialization import safe_globals
from numpy.dtypes import Float64DType, Int64DType
import numpy
from builtins import getattr
from mmdet.visualization import DetLocalVisualizer
import mmcv
import glob
import os


cfg = Config.fromfile('configs/resnetx_mask_rcnn.py')
cfg.test_pipeline = [dict(type='LoadImageFromFile'),
                     dict(type='Resize', scale=(600, 600), keep_ratio=True),
                     dict(type='PackDetInputs')]
with safe_globals([
    HistoryBuffer,
    numpy._core.multiarray._reconstruct,
    numpy._core.multiarray.scalar,
    numpy.ndarray,
    numpy.dtype,
    Float64DType,
    Int64DType,
    getattr
]):
    model = init_detector(cfg, 'work_dirs/resnetx_mask_rcnn/epoch_48.pth', 
                         device='cuda:0')

image_pattern = 'data/raw_frames/*.png'
image_paths = glob.glob(image_pattern)
image_paths.sort()

print(f"Found {len(image_paths)} PNG images to process")

for img_path in image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing: {img_path}")
    
    img = mmcv.imread(img_path)

    result = inference_detector(model, img)

    visualizer = DetLocalVisualizer()
    visualizer.add_datasample(
        'result',
        img,
        result,
        pred_score_thr=0.3,
        out_file=f'preds/train_pred_{img_name}.jpg'
    )