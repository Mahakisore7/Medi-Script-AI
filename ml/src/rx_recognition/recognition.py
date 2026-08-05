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

            # Ensure special tokens are properly configured
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.config.vocab_size = self.model.config.decoder.vocab_size
            self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id

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
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")

            # TrOCR's ViTImageProcessor blindly resizes images to 384x384.
            # The model was trained on text lines that were resized to 384x384, so
            # we should NOT pad it to a square! Padding to a square shrinks the text.

            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=64,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    num_beams=4,
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
_recognizer_finetuned = None
_recognizer_base = None


def recognize_text(image: np.ndarray, use_base_model: bool = False) -> tuple[str, float]:
    """
    Main entry point for pipeline.py.
    """
    global _recognizer_finetuned, _recognizer_base
    import os

    if use_base_model:
        if _recognizer_base is None:
            logging.info("Loading base TrOCR model (microsoft/trocr-base-handwritten)...")
            _recognizer_base = TrOCRRecognizer("microsoft/trocr-base-handwritten")
        return _recognizer_base.recognize_text(image)

    if _recognizer_finetuned is None:
        # Check if the fine-tuned model from Kaggle/Colab has been downloaded
        finetuned_path = "runs/trocr/trocr-medical-best"
        if os.path.exists(finetuned_path):
            logging.info(
                f"✨ Found fine-tuned TrOCR model at {finetuned_path}! Using it for pipeline."
            )
            _recognizer_finetuned = TrOCRRecognizer(finetuned_path)
        else:
            logging.warning(
                f"Fine-tuned model not found at {finetuned_path}. Falling back to base model."
            )
            _recognizer_finetuned = TrOCRRecognizer("microsoft/trocr-base-handwritten")

    return _recognizer_finetuned.recognize_text(image)
