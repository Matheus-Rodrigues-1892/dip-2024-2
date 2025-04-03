# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # 1. Translated image (shift right and down)
    translated = np.zeros_like(img)
    shift_x, shift_y = 10, 10  # Example shift values
    translated[shift_y:, shift_x:] = img[:img.shape[0] - shift_y, :img.shape[1] - shift_x]

    # 2. Rotated image (90 degrees clockwise)
    rotated = np.rot90(img, k=-1)

    # 3. Horizontally stretched image (scale width by 1.5)
    stretched = np.repeat(img, 3 // 2, axis=1)  # Approximation for scaling width by 1.5

    # 4. Horizontally mirrored image (flip along vertical axis)
    mirrored = np.fliplr(img)

    # 5. Barrel distorted image (simple distortion using a radial function)
    distorted = np.zeros_like(img)
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Apply a simple radial distortion
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            factor = 1 + 0.0005 * r**2  # Distortion factor
            src_x = int(center_x + dx / factor)
            src_y = int(center_y + dy / factor)
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                distorted[y, x] = img[src_y, src_x]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }