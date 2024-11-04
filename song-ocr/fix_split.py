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


def DOIT(
    rawPic,
    outdir,
    vSmoothWindow=5,
    hSmoothWindow=5,
    vThreshVal=100,
    maxVThreshVal=500,
    hThreshVal=0.5,
    maxHThreshVal=2.5,
    hMergeThresh=10,
    hSkipThesh=10,
    debug=False,
):
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

    # Apply drip-drop method for vertical segmentation
    V_start, V_end = dripDropMethod(V, vThreshVal)
    if len(V_start) > len(V_end):
        V_end.append(w - 5)

    # Remove segments with width less than 20
    filtered_V_start = []
    filtered_V_end = []
    for i in range(len(V_start)):
        if V_end[i] - V_start[i] >= 5:
            filtered_V_start.append(V_start[i])
            filtered_V_end.append(V_end[i])
    V_start, V_end = filtered_V_start, filtered_V_end

    # Calculate median length of segments
    segment_lengths = [V_end[i] - V_start[i] for i in range(len(V_start))]
    median_length = np.median(segment_lengths)

    # Create directory to save cropped character images
    crop_dir = f"{outdir}/cropped_characters"
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    # Plot vertical projection for visualization
    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(V, label="Smoothed Vertical Projection")
        for start, end in zip(V_start, V_end):
            plt.axvspan(
                start,
                end,
                color="y",
                alpha=0.3,
                label="Vertical Cut" if start == V_start[0] else "",
            )
        plt.title("Vertical Projection")
        plt.xlabel("Column Index")
        plt.ylabel("Number of White Pixels")
        plt.legend()
        plt.savefig(f"{outdir}/vplot.png")
        plt.show()

    # Draw rectangles and perform horizontal segmentation within each vertical segment
    number = 0
    plt.figure(
        figsize=(10, len(V_start) * 8)
    )  # Create a figure for all horizontal projections

    i = 0
    while i < len(V_start):
        # Skip segments that are too narrow (likely incorrect cuts)
        segment_width = V_end[i] - V_start[i]
        if segment_width < median_length * 0.8:
            i += 1
            continue

        # If the segment is too wide, reapply segmentation with increased threshold
        tmpThresh = vThreshVal
        while segment_width > median_length * 1.5:
            tmpThresh += 10
            if tmpThresh > maxVThreshVal:
                break
            sub_V_start, sub_V_end = dripDropMethod(V[V_start[i] : V_end[i]], tmpThresh)

            # Adjust the segment indices relative to the full image
            sub_V_start = [start + V_start[i] for start in sub_V_start]
            sub_V_end = [end + V_start[i] for end in sub_V_end]

            # Replace the current segment with the new sub-segments
            V_start = V_start[:i] + sub_V_start + V_start[i + 1 :]
            V_end = V_end[:i] + sub_V_end + V_end[i + 1 :]

            # Recalculate segment width
            segment_width = V_end[i] - V_start[i]

        rectMin = (V_start[i], 0)
        rectMax = (V_end[i], h)
        cv2.rectangle(origineImage, rectMin, rectMax, (0, 0, 255), 2)

        # Crop the vertical segment
        vertical_segment = img[0:h, V_start[i] : V_end[i]]
        # Horizontal projection
        H, _ = getHProjection(vertical_segment)
        # Smooth horizontal projection
        H = smoothProjection(H, hSmoothWindow)
        # Apply drip-drop method for horizontal segmentation
        H_start, H_end = dripDropMethod(H, hThreshVal)

        # print(f"col: {i}, len(H_start): {len(H_start)}")

        j = 0
        while j < len(H_start):
            segment_height = H_end[j] - H_start[j]
            tmpThresh = hThreshVal
            # print(f"Debug: segment height [{H_start[j]}, {H_end[j]}]")
            while segment_height > median_length * 1.5:
                tmpThresh += 0.1
                # print(
                #     f"Debug: resplit height, current height: {segment_height} [{H_start[j]}, {H_end[j]}], threshold: {tmpThresh}"
                # )
                if tmpThresh >= maxHThreshVal:
                    break
                sub_H_start, sub_H_end = dripDropMethod(
                    H[H_start[j] : H_end[j]], tmpThresh
                )

                # Adjust the segment indices relative to the full image
                sub_H_start = [start + H_start[j] for start in sub_H_start]
                sub_H_end = [end + H_start[j] for end in sub_H_end]

                # Replace the current segment with the new sub-segments
                H_start = H_start[:j] + sub_H_start + H_start[j + 1 :]
                H_end = H_end[:j] + sub_H_end + H_end[j + 1 :]

                # Recalculate segment height
                segment_height = H_end[j] - H_start[j]

            j += 1

        # Remove segments with height less than xx at the beginning and end before merging
        if len(H_start) > 0 and (H_end[0] - H_start[0]) < 20:
            H_start.pop(0)
            H_end.pop(0)
        if len(H_start) > 0 and (H_end[-1] - H_start[-1]) < 20:
            H_start.pop(-1)
            H_end.pop(-1)
        
        # Merge segments that are very close to each other
        merged_H_start = [H_start[0]] if H_start else []
        merged_H_end = []
        for j in range(1, len(H_start)):
            if H_start[j] - H_end[j - 1] < hMergeThresh:
                continue
            merged_H_end.append(H_end[j - 1])
            merged_H_start.append(H_start[j])
        if H_start:
            merged_H_end.append(H_end[-1])
        H_start, H_end = merged_H_start, merged_H_end

        # Draw rectangles on the original image for horizontal segmentation and save cropped characters
        for j in range(len(H_start)):
            if H_end[j] - H_start[j] < hSkipThesh:
                continue

            rectMinH = (V_start[i], H_start[j])
            rectMaxH = (V_end[i], H_end[j])
            cv2.rectangle(origineImage, rectMinH, rectMaxH, (0, 255, 0), 2)

            # Crop character from the original image (not the one with drawn rectangles)
            cropped_char = image[H_start[j] : H_end[j], V_start[i] : V_end[i]]
            char_filename = os.path.join(crop_dir, f"char_{number}.png")
            cv2.imwrite(char_filename, cropped_char)
            number += 1

        # Plot horizontal projection for each vertical segment
        if debug:
            plt.subplot(len(V_start), 1, i + 1)
            plt.plot(H, label=f"Smoothed Horizontal Projection for Column {i + 1}")
            for start, end in zip(H_start, H_end):
                plt.axvspan(
                    start,
                    end,
                    color="y",
                    alpha=0.3,
                    label="Horizontal Cut" if start == H_start[0] else "",
                )
            plt.title(f"Horizontal Projection for Column {i + 1}")
            plt.xlabel("Row Index")
            plt.ylabel("Number of White Pixels")
            plt.legend()

        i += 1

    # Save the result image
    cv2.imwrite(f"{outdir}/result.jpg", origineImage)
    if debug:
        # plt.tight_layout()
        plt.savefig(f"{outdir}/hplots.png")
        plt.show()
    print(f"Result image saved as '{outdir}/result.jpg'")
    print(f"Cropped character images saved in '{crop_dir}'")
