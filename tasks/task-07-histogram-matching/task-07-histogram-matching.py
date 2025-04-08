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
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Função de histogram matching (já implementada anteriormente)
def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img, dtype=np.uint8)

    for channel in range(3):  # Iterate over RGB channels
        source_channel = source_img[:, :, channel].ravel()
        reference_channel = reference_img[:, :, channel].ravel()

        source_hist, _ = np.histogram(source_channel, bins=256, range=(0, 256), density=True)
        reference_hist, _ = np.histogram(reference_channel, bins=256, range=(0, 256), density=True)

        source_cdf = np.cumsum(source_hist)
        reference_cdf = np.cumsum(reference_hist)

        source_cdf_normalized = (source_cdf * 255).astype(np.uint8)
        reference_cdf_normalized = (reference_cdf * 255).astype(np.uint8)

        mapping = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and reference_cdf_normalized[ref_idx] < source_cdf_normalized[src_idx]:
                ref_idx += 1
            mapping[src_idx] = ref_idx

        matched_channel = mapping[source_channel]
        matched_img[:, :, channel] = matched_channel.reshape(source_img[:, :, channel].shape)

    return matched_img

# Função para aplicar o histogram matching usando scikit-image
def match_histograms_scikit(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """
    Matches histograms using scikit-image's match_histograms function.

    Args:
        source_img (np.ndarray): Source image (H, W, 3) in RGB format.
        reference_img (np.ndarray): Reference image (H, W, 3) in RGB format.

    Returns:
        np.ndarray: Image with matched histograms (H, W, 3) in RGB format.
    """
    return match_histograms(source_img, reference_img).astype(np.uint8)

# Carregar as imagens
source_img = cv.imread("source.jpg")
reference_img = cv.imread("reference.jpg")
output_img = cv.imread("output.jpg")

# Converter para RGB (caso estejam em BGR)
source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)
output_img = cv.cvtColor(output_img, cv.COLOR_BGR2RGB)

# Gerar a imagem transformada
generated_img = match_histograms_rgb(source_img, reference_img)

# Salvar a imagem gerada
generated_output_path = "generated_output.jpg"
cv.imwrite(generated_output_path, cv.cvtColor(generated_img, cv.COLOR_RGB2BGR))  # Converter de RGB para BGR antes de salvar
print(f"Generated image saved as {generated_output_path}")

# Gerar a imagem transformada usando scikit-image
generated_img_scikit = match_histograms_scikit(source_img, reference_img)

# Salvar a imagem gerada pelo scikit-image
generated_scikit_output_path = "generated_output_scikit.jpg"
cv.imwrite(generated_scikit_output_path, cv.cvtColor(generated_img_scikit, cv.COLOR_RGB2BGR))  # Converter de RGB para BGR antes de salvar
print(f"Generated image using scikit-image saved as {generated_scikit_output_path}")

# Função para plotar histogramas
def plot_histograms_comparison(images, titles, output_file):
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(15, 12))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 2, i + 1)
        for j, color in enumerate(colors):
            hist, bins = np.histogram(img[:, :, j], bins=256, range=(0, 256))
            plt.plot(bins[:-1], hist, color=color, label=f'{color.upper()} Channel')
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Plotar e salvar os histogramas comparando todas as imagens
images = [source_img, reference_img, output_img, generated_img, generated_img_scikit]
titles = ['Source Image', 'Reference Image', 'Output Image', 'Generated Image (Custom)', 'Generated Image (Scikit-Image)']
plot_histograms_comparison(images, titles, "histograms_comparison_with_scikit.png")