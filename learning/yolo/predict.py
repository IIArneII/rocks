from ultralytics import YOLO
import cv2
import numpy as np

DATASET_CONFIG_PATH = "workspace/yolo/yolo.yaml"
MODEL_NAME = "output/yolo/custom_training4/weights/best.pt"

def predict_yolo():
    model = YOLO(MODEL_NAME)
    results = model.predict('data/yolo/images/val/249.png')

    for i, result in enumerate(results):
        img = result.orig_img.copy()

        masks = result.masks.data.cpu().numpy()

        colored = np.zeros_like(img)
        for j, mask in enumerate(masks):
            m = mask.astype(np.uint8)
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            colored[m] = color

        alpha = 0.5
        overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)

        cv2.imwrite(f'data/yolo/images/test_{i}.png', overlay)

if __name__ == "__main__":
    predict_yolo()
