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

    def recognize_text(self, image: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            return "dummy_text", 50.0

        try:
            # Convert NumPy array (BGR/RGB) to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    pass
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")

            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values, 
                    return_dict_in_generate=True, 
                    output_scores=True
                )

            generated_ids = outputs.sequences
            
            # Compute confidence score from transition scores (requires transformers support)
            try:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                # Average the token probabilities to get a pseudo-confidence score (0-100)
                # Ensure we handle empty sequences
                if transition_scores.numel() > 0:
                    token_probs = torch.exp(transition_scores[0])
                    # Filter out NaN/Inf just in case
                    token_probs = token_probs[torch.isfinite(token_probs)]
                    if token_probs.numel() > 0:
                        avg_prob = token_probs.mean().item()
                        conf = float(avg_prob) * 100.0
                    else:
                        conf = 50.0
                else:
                    conf = 50.0
            except Exception as e:
                logging.warning(f"Failed to compute transition scores: {e}")
                conf = 50.0

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip(), conf
        except Exception as e:
            logging.error(f"TrOCR recognition error: {e}")
            return "", 0.0


# We keep a singleton for the pipeline so it doesn't reload the 1.3GB model every call!
_recognizer = None


def recognize_text(image: np.ndarray) -> tuple[str, float]:
    """
    Main entry point for pipeline.py.
    """
    global _recognizer
    if _recognizer is None:
        # NOTE: Once you fine-tune the model on Kaggle, change this path to "./trocr-rx-finetuned"
        _recognizer = TrOCRRecognizer("microsoft/trocr-base-handwritten")

    return _recognizer.recognize_text(image)
