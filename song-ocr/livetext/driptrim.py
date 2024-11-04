import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

def getVProjection(image):
    """
    Calculate vertical projection, suitable for segmenting vertical text.
    """
    (h, w) = image.shape
    w_ = [0] * w
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    return w_

def getHProjection(image):
    """
    Calculate horizontal projection, suitable for segmenting text rows within a column.
    """
    (h, w) = image.shape
    h_ = [0] * h
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    return h_

def smoothProjection(projection, window_size=5):
    """
    Smooth the projection data using a uniform filter.
    """
    return uniform_filter1d(projection, size=window_size)

def dripDropMethod(projection, threshold=1):
    """
    Apply the drip-drop method to find start and end positions in the projection.
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

def drip_water_cut(image, vSmoothWindow=5, hSmoothWindow=5, vThreshVal=1, hThreshVal=4):
    """
    Perform a drip water cut on the given image to remove unnecessary whitespace using projection analysis.

    Parameters:
    image (PIL.Image): The input image to be processed.
    vSmoothWindow (int): Smoothing window size for vertical projection.
    hSmoothWindow (int): Smoothing window size for horizontal projection.
    vThreshVal (int): Threshold value for vertical drip segmentation.
    hThreshVal (int): Threshold value for horizontal drip segmentation.

    Returns:
    PIL.Image: The cropped image with unnecessary whitespace removed.
    """
    # Convert the image to grayscale and then to binary (black and white)
    gray_img = np.array(image.convert('L'))
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Get vertical and horizontal projections
    v_projection = getVProjection(binary_img)
    h_projection = getHProjection(binary_img)

    # Smooth projections
    v_projection = smoothProjection(v_projection, vSmoothWindow)
    h_projection = smoothProjection(h_projection, hSmoothWindow)

    # Apply drip-drop method to find start and end positions
    v_start, v_end = dripDropMethod(v_projection, vThreshVal)
    h_start, h_end = dripDropMethod(h_projection, hThreshVal)

    # Determine cropping box
    if v_start and h_start:
        left = v_start[0]
        right = v_end[-1]
        top = h_start[0]
        bottom = h_end[-1]
        return image.crop((left, top, right + 1, bottom + 1))
    else:
        # If no valid content is found, return the original image
        return image