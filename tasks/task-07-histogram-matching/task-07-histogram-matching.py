# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """
    Matches the histogram of each RGB channel of the source image to the reference image.

    Args:
        source_img (np.ndarray): Source image as a NumPy array (H, W, 3).
        reference_img (np.ndarray): Reference image as a NumPy array (H, W, 3).

    Returns:
        np.ndarray: Image with matched histograms as a NumPy array (uint8).
    """
    # Ensure the input images are in RGB format
    if source_img.shape[-1] != 3 or reference_img.shape[-1] != 3:
        raise ValueError("Both source and reference images must be in RGB format.")

    # Initialize an empty array for the matched image
    matched_img = np.zeros_like(source_img, dtype=np.uint8)

    # Match histograms for each channel (R, G, B)
    for channel in range(3):
        matched_img[..., channel] = match_histograms(
            source_img[..., channel], reference_img[..., channel]
        ).astype(np.uint8)

    return matched_img

# Load the source and reference images
source_path = r"C:\Users\mathe\Ufal\dip-2024-2\tasks\task-07-histogram-matching\source.jpg"
reference_path = r"C:\Users\mathe\Ufal\dip-2024-2\tasks\task-07-histogram-matching\reference.jpg"
output_path = r"C:\Users\mathe\Ufal\dip-2024-2\tasks\task-07-histogram-matching\output.jpg"

# OpenCV loads images in BGR format, so we convert them to RGB
source_img = cv.imread(source_path)
reference_img = cv.imread(reference_path)

if source_img is None or reference_img is None:
    raise FileNotFoundError("Source or reference image not found. Check the file paths.")

source_img_rgb = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
reference_img_rgb = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)

# Perform histogram matching
matched_img_rgb = match_histograms_rgb(source_img_rgb, reference_img_rgb)

# Convert the matched image back to BGR for saving with OpenCV
matched_img_bgr = cv.cvtColor(matched_img_rgb, cv.COLOR_RGB2BGR)

# Save the output image
cv.imwrite(output_path, matched_img_bgr)

# Optional: Plot histograms of the original and matched images
def plot_histograms(image, title):
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(colors):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()

# Plot histograms for source, reference, and matched images
plot_histograms(source_img_rgb, "Source Image Histograms")
plot_histograms(reference_img_rgb, "Reference Image Histograms")
plot_histograms(matched_img_rgb, "Matched Image Histograms")