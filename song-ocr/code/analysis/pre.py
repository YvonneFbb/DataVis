#!/usr/bin/env python3
import os
import cv2
import numpy as np
import json
import click

def get_image_paths_from_folder(folder_path):
    """
    Get all image paths from a folder.
    :param folder_path: Path to the folder
    :return: List of all image paths
    """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if os.path.splitext(filename)[1].lower() in supported_extensions
    ]
    return image_paths

def get_image_properties(image_path):
    """
    Get the width, height, and number of dark pixels of an image.
    :param image_path: Path to the image
    :return: A tuple containing width, height, and number of dark pixels
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    height, width = image.shape[:2]
    # Convert image to grayscale before processing dark pixels
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Consider a pixel as dark if its value is below a certain threshold (e.g., 10 out of 255)
    dark_threshold = 10
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    dark_pixels = int(np.sum(binary_image < dark_threshold))
    return width, height, dark_pixels

def process_images_in_folder(folder_path):
    """
    Process all images in a folder to get their properties.
    :param folder_path: Path to the folder
    :return: A list of dictionaries containing width, height, and number of dark pixels for each image
    """
    image_paths = get_image_paths_from_folder(folder_path)
    results = []
    for image_path in image_paths:
        try:
            width, height, dark_pixels = get_image_properties(image_path)
            results.append({
                'image_path': image_path,
                'width': width,
                'height': height,
                'dark_pixels': dark_pixels
            })
        except ValueError as e:
            print(e)
    return results

def save_results_to_json(results, folder_path):
    """
    Save the results to a JSON file.
    :param results: List of dictionaries containing image properties
    :param folder_path: Path to the folder where the images are located
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_path = f"{folder_name}_image_properties.json"
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

@click.command()
@click.argument('folder_path', type=click.Path(exists=True))
def main(folder_path):
    results = process_images_in_folder(folder_path)
    if results:
        save_results_to_json(results, folder_path)
        folder_name = os.path.basename(os.path.normpath(folder_path))
        output_path = f"{folder_name}_image_properties.json"
        print(f"Results saved to {output_path}")
    else:
        print("No valid images found or unable to process images.")

if __name__ == "__main__":
    main()
