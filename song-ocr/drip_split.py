import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_border(image):
    """
    Detect and remove the border of the image to allow proper drip segmentation.
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, assuming it is the border
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to remove the border
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image, (x, y, w, h)

def drip_water_segmentation(image):
    """
    Perform character segmentation using the Drip Water Method.
    """
    # Remove the border to ensure proper dripping
    cropped_image, bounding_box = remove_border(image)

    # Show the border removal result
    plt.figure(figsize=(10, 6))
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Border Removed Image')
    plt.savefig('./border_removed.png')
    plt.show()

    # Convert to grayscale if the image is in color
    if len(cropped_image.shape) == 3:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image
    _, binary = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Get image dimensions
    height, width = binary.shape
    
    # Initialize an array to track the drip paths
    drip_paths = np.zeros_like(binary, dtype=np.uint8)

    # Simulate water dripping from top to bottom for each column
    for col in range(width):
        for row in range(height):
            if binary[row, col] == 0:  # If the current pixel is background (black)
                drip_paths[row, col] = 255
            else:
                break  # Stop dripping when hitting a character (white pixel)
    
    # Show the drip paths for visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(drip_paths, cmap='gray')
    plt.title('Drip Paths Visualization')
    plt.savefig('./drip_paths.png')
    plt.show()
    
    # Sum up the drip paths to determine the regions with gaps between characters
    column_sums = np.sum(drip_paths == 255, axis=0)
    
    # Plot the column sums for visualization
    plt.figure(figsize=(10, 4))
    plt.plot(column_sums, label='Column Sums (Drip Paths)')
    plt.title('Drip Water Method - Column Sums')
    plt.xlabel('Column Index')
    plt.ylabel('Drip Path Count')
    plt.legend()
    plt.savefig('./column_sums.png')
    plt.show()
    
    # Detect valleys in the column sums to determine character boundaries
    threshold = 10  # Minimum value to consider a column as a gap
    segment_positions = []
    in_gap = False
    for col in range(width):
        if column_sums[col] >= threshold and not in_gap:
            # Start of a new gap
            segment_positions.append(col)
            in_gap = True
        elif column_sums[col] < threshold and in_gap:
            # End of a gap
            segment_positions.append(col)
            in_gap = False

    # Ensure that we have an even number of segment positions
    if len(segment_positions) % 2 != 0:
        segment_positions.append(width - 1)
    
    # Draw segmentation lines on the original image
    segmented_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(segment_positions), 2):
        start_col = segment_positions[i]
        end_col = segment_positions[i + 1]
        cv2.rectangle(segmented_image, (start_col, 0), (end_col, height), (0, 0, 255), 2)
    
    # Save and display the result
    cv2.imwrite('./drip_result.jpg', segmented_image)
    plt.imshow(segmented_image)
    plt.title('Drip Water Segmentation Result')
    plt.savefig('./drip_segmented.png')
    plt.show()

# Test image path
image_path = './test.png'
image = cv2.imread(image_path)
drip_water_segmentation(image)