import cv2
import numpy as np
import os

def increase_contrast(img):
    """Повышает контрастность изображения через HSV."""
    # Конвертация в HSV и увеличение контраста V-канала
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Применение CLAHE (адаптивное повышение контраста)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    hsv = cv2.merge([h, s, v])
    img_contrast = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_contrast

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """Применяет билатериальный фильтр для сглаживания с сохранением границ."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def process_image(input_path, output_dir):
    """Обрабатывает изображение: контраст + билатериальный фильтр."""
    # Проверяем существование директории, создаем если нужно
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем изображение
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Изображение по пути {input_path} не найдено.")
    
    # Увеличиваем контрастность
    contrasted_img = increase_contrast(img)
    
    # Сохраняем изображение с повышенной контрастностью
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    contrast_path = os.path.join(output_dir, f"{base_name}_contrasted.jpg")
    cv2.imwrite(contrast_path, contrasted_img)
    print(f"Изображение с контрастом сохранено: {contrast_path}")
    
    # Применяем билатериальный фильтр
    filtered_img = apply_bilateral_filter(contrasted_img)
    
    # Сохраняем отфильтрованное изображение
    filtered_path = os.path.join(output_dir, f"{base_name}_bilateral.jpg")
    cv2.imwrite(filtered_path, filtered_img)
    print(f"Изображение после билатериального фильтра сохранено: {filtered_path}")

if __name__ == "__main__":
    # Пример использования (замените пути на свои)
    input_image_path = "images/selected_regions/181.png"  # Путь к исходному изображению
    output_directory = "proc_images"    # Директория для сохранения
    
    process_image(input_image_path, output_directory)
