import os
from pathlib import Path
import shutil


def process_pingpong_images():
    # Get all image files from the Img directory
    img_dir = Path("img-original")
    output_dir = Path("sd-dataset-hyperNetwork")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get all image files
    image_files = [f for f in img_dir.glob("img_*_*.png")]

    # Process each image and create prompts
    prompt_entries = []
    for img_file in image_files:
        # Extract numbers from filename (img_0_4.png -> [0, 4])
        name_parts = img_file.stem.split('_')
        serve_time = int(name_parts[1])
        zone_index = int(name_parts[2])


        # Create new filename
        new_filename = f"{serve_time}-serve-which-pingpong-ball-landed-in-zone-{zone_index}.png"

        # Copy and rename the file
        shutil.copy2(img_file, output_dir / new_filename)


    print(f"Copied and renamed {len(prompt_entries)} images to {output_dir}")


if __name__ == "__main__":
    try:
        process_pingpong_images()
    except FileNotFoundError:
        print("Error: 'Img' directory not found. Please make sure it exists in the current working directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")