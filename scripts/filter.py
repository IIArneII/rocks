import json
import os
import shutil

input_json = 'data/instances.json'
output_json = 'data/30/instances.json'
source_images_dir = 'images/selected_regions'
target_images_dir = 'data/30/images'
allowed_image_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 38, 39, 40, 43, 44, 49, 51}

os.makedirs(target_images_dir, exist_ok=True)

with open(input_json, 'r') as f:
    coco_data = json.load(f)

filtered_images = [img for img in coco_data['images'] if img['id'] in allowed_image_ids]

filtered_image_ids = {img['id'] for img in filtered_images}

filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in filtered_image_ids]

for img in filtered_images:

    source_file = os.path.join(source_images_dir, img['file_name'])
    target_file = os.path.join(target_images_dir, img['file_name'])

    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)
        print(f"Скопировано: {source_file} -> {target_file}")
    else:
        print(f"Предупреждение: файл не найден {source_file}")

filtered_coco_data = {
    "info": coco_data.get('info', {}),
    "licenses": coco_data.get('licenses', []),
    "categories": coco_data.get('categories', []),
    "images": filtered_images,
    "annotations": filtered_annotations
}

with open(output_json, 'w') as f:
    json.dump(filtered_coco_data, f, indent=2)

print(f"\nФайл {output_json} успешно создан")
print(f"Изображения скопированы в {target_images_dir}")
