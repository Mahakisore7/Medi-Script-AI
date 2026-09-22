import logging

import numpy as np
import torch
from PIL import Image

try:
    import transformers  # noqa: F401 — availability check
except ImportError:
    logging.warning("Transformers not installed; TrOCR will fail.")


class TrOCRRecognizer:
    def __init__(self, model_path="models/final_trocr_large_model"):
        """
        Loads the locally fine-tuned TrOCR-Large model and processor.
        Ensure you have extracted the downloaded Colab model
        to models/final_trocr_large_model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import (
                RobertaTokenizer,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
            )

            # Load components manually to avoid FastTokenizer bugs in HuggingFace
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

            # Clear max_length from generation_config to avoid conflict with max_new_tokens
            if (
                hasattr(self.model, "generation_config")
                and self.model.generation_config is not None
            ):
                self.model.generation_config.max_length = None

            logging.info("TrOCR loaded successfully.")

            # Cache for prefix allowed tokens
            self._token_str_list = []
            for i in range(self.processor.tokenizer.vocab_size):
                s = self.processor.tokenizer.decode([i], skip_special_tokens=True).lower()
                if s:
                    self._token_str_list.append((i, s))

            from rx_recognition.lexicon_trie import MedicalLexicons

            self.lexicons = MedicalLexicons()

        except Exception as e:
            logging.error(f"Failed to load TrOCR: {e}")
            self.model = None

    def recognize_text(self, image: np.ndarray, field_type: str = None) -> tuple[str, float]:
        if self.model is None:
            return "dummy_text", 50.0

        try:
            # Convert NumPy array (BGR/RGB) to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = image.convert("RGB")

            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(
                self.device
            )

            # NOTE: prefix_allowed_tokens_fn is disabled on CPU — it iterates the
            # entire vocab every token and makes inference 100x slower.
            # The fine-tuned model + fuzzy lexicon handles correction instead.
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=24,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    num_beams=2,
                    num_return_sequences=2 if field_type == "medicine" else 1,
                )

            beam_results = []
            for i in range(outputs.sequences.shape[0]):
                seq = outputs.sequences[i]
                seq_text = self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()

                # Compute conf for this sequence
                try:
                    if outputs.scores:
                        transition_scores = self.model.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )
                        if transition_scores.numel() > i:
                            token_probs = torch.exp(transition_scores[i])
                            token_probs = token_probs[torch.isfinite(token_probs)]
                            conf = (
                                float(token_probs.mean().item()) * 100.0
                                if token_probs.numel() > 0
                                else 50.0
                            )
                        else:
                            conf = 50.0
                    else:
                        conf = 50.0
                except Exception:
                    conf = 50.0

                beam_results.append((seq_text, conf))

            from rx_recognition.domain_lm import get_domain_lm

            lm = get_domain_lm()
            best_text, best_conf = lm.rescore_beams(beam_results, field_type=field_type)

            return best_text.strip(), best_conf
        except Exception as e:
            logging.error(f"TrOCR recognition error: {e}")
            return "", 0.0


# We keep a singleton for the pipeline so it doesn't reload the 1.3GB model every call!
_recognizer_finetuned = None
_recognizer_base = None


def recognize_text(
    image: np.ndarray, use_base_model: bool = False, field_type: str = None
) -> tuple[str, float]:
    """
    Main entry point for pipeline.py.
    """
    global _recognizer_finetuned, _recognizer_base
    import os

    if use_base_model:
        if _recognizer_base is None:
            logging.info("Loading base TrOCR model (microsoft/trocr-large-handwritten)...")
            _recognizer_base = TrOCRRecognizer("microsoft/trocr-large-handwritten")
        return _recognizer_base.recognize_text(image, field_type=field_type)

    if _recognizer_finetuned is None:
        # Check if the fine-tuned model from Kaggle/Colab has been downloaded
        local_model_v4 = "models/final_trocr_large_model_v4_dosage"
        local_model_v3 = "models/final_trocr_large_model_v3"
        local_model_v2 = "models/final_trocr_large_model_v2"
        local_model_v1 = "models/final_trocr_large_model"

        if os.path.exists(local_model_v4):
            logging.info(f"Using fine-tuned TrOCR model (v4) from {local_model_v4}")
            _recognizer_finetuned = TrOCRRecognizer(local_model_v4)
        elif os.path.exists(local_model_v3):
            logging.info(f"Using fine-tuned TrOCR model (v3) from {local_model_v3}")
            _recognizer_finetuned = TrOCRRecognizer(local_model_v3)
        elif os.path.exists(local_model_v2):
            logging.info(f"Using fine-tuned TrOCR model (v2) from {local_model_v2}")
            _recognizer_finetuned = TrOCRRecognizer(local_model_v2)
        elif os.path.exists(local_model_v1):
            logging.info(f"Using fine-tuned TrOCR model (v1) from {local_model_v1}")
            _recognizer_finetuned = TrOCRRecognizer(local_model_v1)
        else:
            logging.warning(
                "No local models found! Falling back to microsoft/trocr-large-handwritten"
            )
            _recognizer_finetuned = TrOCRRecognizer("microsoft/trocr-large-handwritten")

    return _recognizer_finetuned.recognize_text(image, field_type=field_type)
