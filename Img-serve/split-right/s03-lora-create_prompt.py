import os
from pathlib import Path
import shutil


def process_pingpong_images():
    # Get all image files from the Img directory
    img_dir = Path("Img-original")
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


        # Create new filename base (without extension)
        new_filename_base = f"{serve_time}-serve-which-pingpong-ball-landed-in-zone-{zone_index}"

        # Copy and rename the image file
        shutil.copy2(img_file, output_dir / f"{new_filename_base}.png")

        # Create detailed prompt for individual txt file
        detailed_prompt = f"pingpong serve, table, pingpong table, white pingpong ball,no human,ball landed on zone {zone_index} of left pingpong table"

        # Create matching txt file
        txt_path = output_dir / f"{new_filename_base}.txt"
        with open(txt_path, 'w') as f:
            f.write(detailed_prompt)


    print(f"Created {len(prompt_entries)} image-text pairs in {output_dir}")


if __name__ == "__main__":
    try:
        process_pingpong_images()
    except FileNotFoundError:
        print("Error: 'Img' directory not found. Please make sure it exists in the current working directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")