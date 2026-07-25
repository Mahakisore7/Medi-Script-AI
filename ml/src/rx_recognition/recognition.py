import logging

import numpy as np
import pytesseract


def recognize_text(image: np.ndarray) -> str:
    """
    Placeholder recognition module using Tesseract.
    This will be replaced by TrOCR in Phase 4.
    """
    try:
        # psd 7 = Treat the image as a single text line.
        text = pytesseract.image_to_string(image, config="--psm 7").strip()
        return text
    except pytesseract.TesseractNotFoundError:
        logging.warning("Tesseract not installed on system. Returning dummy text.")
        return "dummy_text"
    except Exception as e:
        logging.error(f"OCR Error: {e}")
        return ""
