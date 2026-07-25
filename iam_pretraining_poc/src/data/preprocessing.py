import cv2
import numpy as np
from PIL import Image

def deskew(image):
    """
    Deskews the image by detecting the text orientation and rotating it.
    """
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def preprocess_image(img_path):
    """
    Applies standard preprocessing steps to a prescription image:
    Grayscale, binarization, and deskewing.
    Returns a PIL Image ready for the TrOCR Processor.
    """
    # 1. Read in grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    # 2. Contrast normalization
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Adaptive binarization (Gaussian)
    # Using adaptive thresholding helps with varying lighting in real prescriptions
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 4. Deskewing
    deskewed = deskew(thresh)
    
    # Invert back to normal (black text on white background)
    final = cv2.bitwise_not(deskewed)
    
    # Convert back to PIL Image (TrOCR expects PIL Images or numpy arrays in RGB)
    # TrOCR Processor handles the final normalization and resizing
    final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(final_rgb)
    
    return pil_image
