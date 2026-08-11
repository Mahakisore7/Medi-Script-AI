"""
crop_enhancer.py
Pre-processes YOLO crop images before feeding to EasyOCR.
Applying these transforms dramatically improves EasyOCR accuracy on
small, low-contrast handwritten text crops.
"""

import cv2
import numpy as np


def enhance_crop_for_ocr(crop: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """
    Enhances a YOLO crop image for better OCR accuracy.
    Steps:
      1. Upscale (handwriting is usually small in the crop)
      2. Convert to grayscale
      3. Denoise
      4. Adaptive thresholding (binarize — works better than global threshold on uneven lighting)
      5. Dilate slightly to thicken thin strokes
    """
    if crop is None or crop.size == 0:
        return crop

    # 1. Upscale — EasyOCR works much better on larger images
    h, w = crop.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Grayscale
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #    Boosts local contrast without blowing out highlights
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    contrast = clahe.apply(denoised)

    # 5. Adaptive thresholding — binarize to pure black/white
    binary = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )

    # 6. Slight dilation to thicken pen strokes (helps with thin handwriting)
    kernel = np.ones((2, 2), np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=1)

    # Convert back to BGR so EasyOCR can accept it
    result = cv2.cvtColor(thickened, cv2.COLOR_GRAY2BGR)
    return result
