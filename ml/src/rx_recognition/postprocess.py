from rapidfuzz import fuzz, process

from rx_recognition.lexicon import DRUG_LEXICON


def postprocess_text(field_name: str, raw_text: str) -> str:
    """
    Applies domain-specific postprocessing (Phase 5).
    For the medicine field, we fuzzy match against a known drug lexicon
    to correct minor OCR spelling mistakes.
    """
    if field_name == "medicine" and raw_text.strip():
        # Rapidfuzz extractOne returns (match, score, index)
        match = process.extractOne(raw_text, DRUG_LEXICON, scorer=fuzz.WRatio)
        if match and match[1] > 60:  # Threshold
            return match[0]

    return raw_text.strip()
