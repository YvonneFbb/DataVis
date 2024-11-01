import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def getVProjection(image):
    """
    Calculate vertical projection, suitable for segmenting vertical text.
    """
    vProjection = np.zeros(image.shape, np.uint8)
    (h, w) = image.shape
    w_ = [0] * w
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # Draw vertical projection image (for visualization)
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    return w_, vProjection

def getHProjection(image):
    """
    Calculate horizontal projection, suitable for segmenting text rows within a column.
    """
    hProjection = np.zeros(image.shape, np.uint8)
    (h, w) = image.shape
    h_ = [0] * h
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # Draw horizontal projection image (for visualization)
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    return h_, hProjection

def smoothProjection(projection, window_size=5):
    """
    Smooth the projection data using a uniform filter.
    """
    return uniform_filter1d(projection, size=window_size)

def getOptimalThreshold(vProjection, iterations=3, maxLimitRatio=0.1):
    """
    Calculate an optimal threshold based on the vertical projection.
    Iteratively find local minima for a given number of iterations.
    Ensure the final threshold is below 100.
    """
    current_values = vProjection
    current_indices = list(range(len(vProjection)))

    for iteration in range(iterations):
        # Find peaks and valleys (local minima)
        peaks, _ = find_peaks(-np.array(current_values))

        # Get the values at the valleys (local minima)
        valley_values = [current_values[i] for i in peaks if current_values[i]]
        valley_indices = [current_indices[i] for i in peaks if current_values[i]]

        if len(valley_values) == 0:
            raise ValueError(f"Failed to find sufficient local minima at iteration {iteration + 1}. Ensure the image has appropriate segmentation features.")

        current_values = valley_values
        current_indices = valley_indices

    # Ensure the value is below a given ratio
    maxLimit = maxLimitRatio * max(vProjection)
    final_values = [val for val in current_values if val < maxLimit]
    if len(final_values) == 0:
        raise ValueError("No suitable minima found below the threshold of 100. Adjust the criteria or ensure the image quality is appropriate.")

    iog = max(final_values)
    return iog, current_indices

def dripDropMethod(projection, threshold=1):
    """
    Apply the drip-drop method to find start and end positions in the horizontal projection.
    Use a threshold to control the start detection of a new segment.
    """
    start_positions = []
    end_positions = []
    is_segment = False

    for i in range(len(projection)):
        if projection[i] > threshold and not is_segment:
            start_positions.append(i)
            is_segment = True
        elif projection[i] <= threshold and is_segment:
            end_positions.append(i)
            is_segment = False

    if is_segment:
        end_positions.append(len(projection) - 1)

    return start_positions, end_positions

def scan(vProjection, iog, pos=0):
    """
    Scan the projection to find start and end positions with a given threshold.
    """
    start = 0
    V_start = []
    V_end = []

    for i in range(len(vProjection)):
        if vProjection[i] > iog and start == 0:
            V_start.append(i)
            start = 1
        if vProjection[i] <= iog and start == 1:
            if i - V_start[-1] < pos:
                continue
            V_end.append(i)
            start = 0
    return V_start, V_end

def DOIT(rawPic, outdir, vSmoothWindow=5, hSmoothWindow=5, vThreshIter=2, hThreashVal=0.5, debug=False):
    # Read the original image
    origineImage = cv2.imread(rawPic)
    # Convert image to grayscale  
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    # Binarize the image
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # Image height and width
    (h, w) = img.shape
    # Vertical projection
    V, vProjectionImage = getVProjection(img)
    # Smooth vertical projection
    V = smoothProjection(V, vSmoothWindow)
    
    # Get optimal threshold for vertical scanning
    try:
        iog, refined_peaks = getOptimalThreshold(V, vThreshIter)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Split based on vertical projection
    V_start, V_end = scan(V, iog)  # Use the calculated optimal threshold
    if len(V_start) > len(V_end):
        V_end.append(w - 5)

    # Create directory to save cropped character images
    crop_dir = f'{outdir}/cropped_characters'
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    # Plot vertical projection for visualization with refined peaks
    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(V, label='Smoothed Vertical Projection')
        plt.plot(refined_peaks, [V[i] for i in refined_peaks], "rx", label='Refined Local Minima')
        for start, end in zip(V_start, V_end):
            plt.axvspan(start, end, color='y', alpha=0.3, label='Vertical Cut' if start == V_start[0] else '')
        plt.title('Vertical Projection')
        plt.xlabel('Column Index')
        plt.ylabel('Number of White Pixels')
        plt.legend()
        plt.savefig(f'{outdir}/vplot.png')
        plt.show()

    # Draw rectangles and perform horizontal segmentation within each vertical segment
    number = 0
    plt.figure(figsize=(10, len(V_start) * 4))  # Create a figure for all horizontal projections
    for i in range(len(V_start)):
        # Skip segments that are too narrow (likely incorrect cuts)
        if V_end[i] - V_start[i] < 30:  # Threshold can be adjusted
            continue

        rectMin = (V_start[i], 0)
        rectMax = (V_end[i], h)
        cv2.rectangle(origineImage, rectMin, rectMax, (0, 0, 255), 2)

        # Crop the vertical segment
        vertical_segment = img[0:h, V_start[i]:V_end[i]]
        # Horizontal projection
        H, _ = getHProjection(vertical_segment)
        # Smooth horizontal projection
        H = smoothProjection(H, hSmoothWindow)
        # Apply drip-drop method for horizontal segmentation
        H_start, H_end = dripDropMethod(H, hThreashVal)
        # Draw rectangles on the original image for horizontal segmentation and save cropped characters
        for j in range(len(H_start)):
            rectMinH = (V_start[i], H_start[j])
            rectMaxH = (V_end[i], H_end[j])
            cv2.rectangle(origineImage, rectMinH, rectMaxH, (0, 255, 0), 2)
            
            # Crop character from the original image (not the one with drawn rectangles)
            cropped_char = image[H_start[j]:H_end[j], V_start[i]:V_end[i]]
            char_filename = os.path.join(crop_dir, f'char_{number}.png')
            cv2.imwrite(char_filename, cropped_char)
            number += 1
            
        # Plot horizontal projection for each vertical segment
        if debug:
            plt.subplot(len(V_start), 1, i + 1)
            plt.plot(H, label=f'Smoothed Horizontal Projection for Column {i + 1}')
            for start, end in zip(H_start, H_end):
                plt.axvspan(start, end, color='y', alpha=0.3, label='Horizontal Cut' if start == H_start[0] else '')
            plt.title(f'Horizontal Projection for Column {i + 1}')
            plt.xlabel('Row Index')
            plt.ylabel('Number of White Pixels')
            plt.legend()

    # Save the result image
    cv2.imwrite(f'{outdir}/result.jpg', origineImage)
    if debug:
        plt.tight_layout()
        plt.savefig('./hplots.png')
        plt.show()
    print(f"Result image saved as '{outdir}/result.jpg'")
    print(f"Cropped character images saved in '{crop_dir}'")