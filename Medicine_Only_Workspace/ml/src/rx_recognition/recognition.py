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
                # The tokenizer files are not saved by the trainer by default, and we didn't change the vocab, 
                # so we can always safely load it from the base model!
                tokenizer=RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten"),
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)

            # Ensure special tokens are properly configured
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.config.vocab_size = self.model.config.decoder.vocab_size
            self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id

            self.model.eval()
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
            
            prefix_fn = None
            if field_type == "medicine":
                med_trie = self.lexicons.get_medicines_trie()
                
                def _allowed_tokens(batch_id, input_ids):
                    text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=True).lower()
                    allowed_token_ids = []
                    
                    if med_trie.is_valid_word(text):
                        allowed_token_ids.append(self.processor.tokenizer.eos_token_id)
                        
                    for token_id, token_str in self._token_str_list:
                        if med_trie.is_valid_prefix(text + token_str):
                            allowed_token_ids.append(token_id)
                            
                    if not allowed_token_ids:
                        allowed_token_ids.append(self.processor.tokenizer.eos_token_id)
                    return allowed_token_ids
                
                prefix_fn = _allowed_tokens

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=32,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    num_beams=4,
                    num_return_sequences=4 if field_type == "medicine" else 1,
                    prefix_allowed_tokens_fn=prefix_fn,
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
                            conf = float(token_probs.mean().item()) * 100.0 if token_probs.numel() > 0 else 50.0
                        else:
                            conf = 50.0
                    else:
                        conf = 50.0
                except Exception as e:
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


def recognize_text(image: np.ndarray, use_base_model: bool = False, field_type: str = None) -> tuple[str, float]:
    """
    Main entry point for pipeline.py.
    """
    global _recognizer_finetuned, _recognizer_base
    import os

    if use_base_model:
        if _recognizer_base is None:
            logging.info("Loading base TrOCR model (microsoft/trocr-base-handwritten)...")
            _recognizer_base = TrOCRRecognizer("microsoft/trocr-base-handwritten")
        return _recognizer_base.recognize_text(image, field_type=field_type)

    if _recognizer_finetuned is None:
        # Check if the fine-tuned model from Kaggle/Colab has been downloaded
        script_dir = os.path.dirname(os.path.abspath(__file__))
        finetuned_path = os.path.abspath(
            os.path.join(script_dir, "..", "..", "..", "runs", "trocr", "trocr-medical-best")
        )
        if os.path.exists(finetuned_path):
            logging.info(
                f"✨ Found fine-tuned TrOCR model at {finetuned_path}! Using it for pipeline."
            )
            _recognizer_finetuned = TrOCRRecognizer(finetuned_path)
        else:
            import warnings
            msg = f"Fine-tuned model not found at {finetuned_path}. Falling back to base model."
            warnings.warn(msg)
            logging.warning(msg)
            _recognizer_finetuned = TrOCRRecognizer("microsoft/trocr-base-handwritten")

    return _recognizer_finetuned.recognize_text(image, field_type=field_type)

