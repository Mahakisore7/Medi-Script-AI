import logging

import numpy as np
import torch
from PIL import Image

try:
    from transformers import (
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
    )
except ImportError:
    logging.warning("Transformers not installed; TrOCR will fail.")


class TrOCRRecognizer:
    def __init__(self, model_path="microsoft/trocr-base-handwritten"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading TrOCR model from {model_path} onto {self.device}...")

        try:
            # We use manual instantiation as done in train_trocr to avoid FastTokenizer bugs
            self.processor = TrOCRProcessor(
                image_processor=ViTImageProcessor.from_pretrained(model_path),
                tokenizer=RobertaTokenizer.from_pretrained(model_path),
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            logging.info("TrOCR loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load TrOCR: {e}")
            self.model = None

    def recognize_text(self, image: np.ndarray) -> str:
        if self.model is None:
            return "dummy_text"

        try:
            # Convert NumPy array (BGR/RGB) to PIL Image
            if isinstance(image, np.ndarray):
                # Assuming image is BGR from OpenCV, convert to RGB
                # Wait, pipeline extracts RGB. Let's just convert safely.
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Often cv2 provides BGR, but we assume RGB or handle generically
                    pass
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")

            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
        except Exception as e:
            logging.error(f"TrOCR recognition error: {e}")
            return ""


# We keep a singleton for the pipeline so it doesn't reload the 1.3GB model every call!
_recognizer = None


def recognize_text(image: np.ndarray) -> str:
    """
    Main entry point for pipeline.py.
    """
    global _recognizer
    if _recognizer is None:
        # NOTE: Once you fine-tune the model on Kaggle, change this path to "./trocr-rx-finetuned"
        _recognizer = TrOCRRecognizer("microsoft/trocr-base-handwritten")

    return _recognizer.recognize_text(image)
