import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Applies basic preprocessing for OCR.
    1. Grayscale
    2. OTSU Thresholding (Binarization)
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh
