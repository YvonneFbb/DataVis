#!/usr/bin/env python3
import os
import click
from PIL import Image
from livetext import AppleLiveTextOCR
from driptrim import drip_water_cut


def process_images_in_directory(input_dir, output_dir, locales=["zh"]):
    # Create OCR object
    ocr = AppleLiveTextOCR(locales=locales)
    print("Apple Live Text Framework loaded")

    # Iterate over each image file in the directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            # Open the image to be recognized
            img = Image.open(input_path)
            ocr_result = ocr.perform_ocr(img)

            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Crop characters based on bounding boxes and save them
            img_width, img_height = img.size
            if ocr_result:
                for idx, char_info in enumerate(ocr_result):
                    if char_info["bounds"]:
                        bounds = char_info["bounds"]
                        # Convert normalized coordinates to pixel coordinates
                        x = bounds["origin"][0] * img_width
                        y = bounds["origin"][1] * img_height
                        width = bounds["size"][0] * img_width
                        height = bounds["size"][1] * img_height
                        # Crop the character from the image
                        cropped_img = img.crop((x, y, x + width, y + height))

                        # Perform drip water cut to remove unnecessary whitespace
                        processed_img = drip_water_cut(cropped_img)

                        # Save the processed character using the text as the filename
                        cropped_filename = os.path.join(
                            output_dir,
                            f"{os.path.splitext(filename)[0]}_{char_info['text']}_{idx}.png",
                        )
                        processed_img.save(cropped_filename)
                        print(f"Saved processed character to {cropped_filename}")
            else:
                # Save the image with 'unrecognized' if no text is recognized
                unrecognized_filename = os.path.join(
                    output_dir, f"{os.path.splitext(filename)[0]}_unrecognized.png"
                )
                img.save(unrecognized_filename)
                print(f"Saved unrecognized image to {unrecognized_filename}")


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--locales",
    default=["zh"],
    multiple=True,
    help="Locales for OCR processing (default: zh).",
)
def main(input_dir, locales):
    """Batch OCR processing of images in a directory."""
    output_dir = f"{input_dir}_ocr"
    process_images_in_directory(input_dir, output_dir, locales)


if __name__ == "__main__":
    main()
