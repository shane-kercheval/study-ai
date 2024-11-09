import os
from PIL import Image
from multiprocessing import Pool, cpu_count

def process_image(file_path: str) -> None:
    # Open the image file
    with Image.open(file_path) as img:
        # Calculate new dimensions (50% of the original)
        new_width = int(img.width * 0.5)
        new_height = int(img.height * 0.5)
        # Resize the image using LANCZOS as the resampling filter
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        # Save the resized image, overwriting the original file
        resized_img.save(file_path, optimize=True)
    print(f"Resized {file_path} to 50% of its original dimensions.")

def reduce_png_size_by_50(directory: str) -> None:
    # Get all .png files in the directory
    png_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
    # Use all available CPU cores
    with Pool(cpu_count()) as pool:
        pool.map(process_image, png_files)

if __name__ == '__main__':
    directory_path = "../obsidian_omscs/CS6200-GIOS/Images"
    reduce_png_size_by_50(directory_path)
