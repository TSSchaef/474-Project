import os
import csv
import random
from PIL import Image
import numpy as np

def convert_csv_to_jpg(input_csv, output_dir, image_size=(28, 28)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_csv, 'r') as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)  # Skip the header row if present
            # Read all rows and randomly select 10
            rows = list(reader)
            selected_rows = random.sample(rows, min(4, len(rows)))

            for i, row in enumerate(selected_rows):
                # Assuming the first column is a label and the rest are pixel values
                label = row[0]
                pixels = list(map(int, row[1:]))
                
                # Invert pixel values (flip black and white)
                inverted_pixels = [255 - p for p in pixels]
                image_array = np.array(inverted_pixels, dtype=np.uint8).reshape(image_size)

                # Create an image from the array and rotate it by 90 degrees
                img = Image.fromarray(image_array, mode='L')  # 'L' for grayscale
                img = img.rotate(90, expand=True)  # Rotate 90 degrees clockwise
                output_path = os.path.join(output_dir, f"image_{i}_label_{label}.jpg")
                img.save(output_path)
                print(f"Saved: {output_path}")

                # Save the label and inverted pixel values to a .txt file
                label_path = os.path.join(output_dir, f"image_{i}_label_{label}.txt")
                with open(label_path, 'w') as label_file:
                    label_file.write(f"Label: {label}\n")
                    label_file.write("Pixels:\n")
                    label_file.write(" ".join(map(str, inverted_pixels)))
                print(f"Saved label and pixels: {label_path}")  

    except Exception as e:
        print(f"Failed to process CSV: {e}")

if __name__ == "__main__":
    input_csv = "/home/tim/Documents/CS474/midterm_proj/data/archive/emnist-balanced-train.csv"
    output_directory = "/home/tim/Documents/CS474/midterm_proj/images"

    if not os.path.isfile(input_csv):  # Ensure input is a file
        print(f"Error: {input_csv} is not a valid file.")
    else:
        convert_csv_to_jpg(input_csv, output_directory)
