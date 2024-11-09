import os
from PIL import Image

def reduce_png_size_by_50(directory: str) -> None:
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PNG image
        if filename.endswith('.png'):
            # Open the image file
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                # Calculate new dimensions (50% of the original)
                new_width = int(img.width * 0.5)
                new_height = int(img.height * 0.5)
                # Resize the image
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                # Save the resized image, overwriting the original file
                resized_img.save(file_path, optimize=True)
            print(f"Resized {filename} to 50% of its original dimensions.")

# Specify the directory containing the .png files
directory_path = "../obsidian_omscs/CS6200-GIOS/Images"
reduce_png_size_by_50(directory_path)
