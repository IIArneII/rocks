import os
import json
from PIL import Image

def crop_images_from_coco(source_dir, result_dir, annotations_path):
    """
    Вырезает области из изображений на основе COCO-разметки
    """
    os.makedirs(result_dir, exist_ok=True)

    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = {ann['image_id']: ann for ann in coco_data['annotations']}

    for image_id, ann in annotations.items():
        img_info = images.get(image_id)
        if not img_info:
            print(f"Изображение с ID {image_id} не найдено в аннотациях")
            continue

        src_path = os.path.join(source_dir, img_info['file_name'])
        dst_path = os.path.join(result_dir, img_info['file_name'])

        if not os.path.exists(src_path):
            print(f"Файл {src_path} не существует")
            continue

        try:
            with Image.open(src_path) as img:
                bbox = ann['bbox']
                x, y, w, h = map(int, bbox)

                if w <= 0 or h <= 0:
                    print(f"Некорректный bbox {bbox} для {src_path}")
                    continue

                cropped = img.crop((x, y, x + w, y + h))

                cropped.save(dst_path)
                print(f"Обработано: {dst_path}")

        except Exception as e:
            print(f"Ошибка при обработке {src_path}: {str(e)}")

if __name__ == "__main__":
    SOURCE_DIR = "images/origin"
    RESULT_DIR = "images/regions"
    ANNOTATIONS_PATH = "data/regions.json"

    crop_images_from_coco(SOURCE_DIR, RESULT_DIR, ANNOTATIONS_PATH)
