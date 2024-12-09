#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path
import click

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../livetext")))

from livetext import AppleLiveTextOCR
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# Create OCR object
ocr = AppleLiveTextOCR(locales=["zh"])
print("Apple Live Text Framework loaded")


# Convert PDF pages to images
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images


# Process OCR for each page of the PDF
def ocr_pdf(pdf_path, target_word="永"):
    images = pdf_to_images(pdf_path)
    found_words = []  # Store results where the target word is found
    results = []  # Store OCR results for all pages

    # Create output folder based on the PDF file name
    pdf_name = Path(pdf_path).stem
    output_folder = Path(pdf_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    for page_num, img in enumerate(images):
        print(f"Processing page {page_num + 1}...")

        # Perform OCR on the image
        result = ocr.perform_ocr(img)
        results.append(result)

        found = False  # Track if any target word is found on this page

        # Track word occurrence and mark them
        for one in result:
            if target_word in one["text"]:
                found = True  # Mark as found
                # Record the word's location and the bounding box for cropping
                bounding_box = one["bounds"]
                word_info = {
                    "pdf_path": str(pdf_path),  # Convert path to string for JSON serialization
                    "page": page_num + 1,
                    "word": one["text"],
                    "bounding_box": bounding_box,
                }
                found_words.append(word_info)

                # Convert image to be editable
                draw = ImageDraw.Draw(img)

                # Draw rectangle on the original image to mark the found word
                img_width, img_height = img.size
                x = bounding_box["origin"][0] * img_width
                y = bounding_box["origin"][1] * img_height
                width = bounding_box["size"][0] * img_width
                height = bounding_box["size"][1] * img_height

                # Draw a red rectangle with a width of 2 pixels
                draw.rectangle(
                    [x, y, x + width, y + height],
                    outline="red",
                    width=2
                )

        # Save the image with marked bounding boxes only if any target word was found
        if found:
            marked_image_path = output_folder / f"{pdf_name}_page_{page_num + 1}_marked.png"
            img.save(marked_image_path)
            print(f"Saved marked image for page {page_num + 1} at: {marked_image_path}")

    # Save the OCR results to JSON
    ocr_results_path = output_folder / f"{pdf_name}_ocr_results.json"
    with open(ocr_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved OCR results to: {ocr_results_path}")

    # Save the found words information to JSON
    found_words_path = output_folder / f"{pdf_name}_found_words.json"
    with open(found_words_path, 'w', encoding='utf-8') as f:
        json.dump(found_words, f, ensure_ascii=False, indent=4)
    print(f"Saved found words to: {found_words_path}")

    return found_words


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--target_word', default='永', help='The target word to search for in the PDF files.')
def main(input_folder, target_word):
    """
    Scan the specified folder for PDF files and process each PDF file
    for OCR and extract target word images.
    """
    # List all PDFs in the input folder
    pdf_files = [f for f in Path(input_folder).iterdir() if f.suffix.lower() == '.pdf']

    if not pdf_files:
        print(f"No PDF files found in the directory: {input_folder}")
        return

    print(f"Found {len(pdf_files)} PDF files in {input_folder}. Processing...")

    summary = []  # List to store summary of all found words across all PDFs

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.name}")
        found_words = ocr_pdf(pdf_file, target_word=target_word)

        # Add found words to the summary list
        for word_info in found_words:
            summary.append(word_info)

    # Save the summary of all found words to a JSON file
    summary_path = Path(input_folder) / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"Saved summary of all found words to: {summary_path}")


if __name__ == "__main__":
    main()
