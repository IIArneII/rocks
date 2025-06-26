from PIL import Image
import os

def resize_images_to_square(input_folder="images/selected_regions/", output_folder="images/selected_regions_resized"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                original_width, original_height = img.size

                max_dim = max(original_width, original_height)
                if img.mode == 'P' or img.mode == 'RGBA':
                    img = img.convert('RGB')

                square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))

                paste_x = (max_dim - original_width) // 2
                paste_y = (max_dim - original_height) // 2

                square_img.paste(img, (paste_x, paste_y))

                output_path = os.path.join(output_folder, filename)
                square_img.save(output_path)
                print(f"Resized and saved: {output_path}")

            except Exception as e:
                print(f"Could not process {filename}: {e}")

if __name__ == "__main__":
    resize_images_to_square()
    print("Image resizing complete.")
