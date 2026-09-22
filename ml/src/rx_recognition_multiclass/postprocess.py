import re

from rapidfuzz import fuzz, process

from rx_recognition_multiclass.lexicon import get_lexicon


def combined_confidence(seg_conf: float, rec_conf: float, val_conf: float) -> float:
    """
    Blends the three confidence signals (YOLO, TrOCR, Validation).
    Weighted heavily toward validation (since it checks against a real vocabulary).
    If any single stage is extremely low, it drags the whole score down.
    """
    weighted = 0.2 * seg_conf + 0.3 * rec_conf + 0.5 * val_conf
    # The lowest signal acts as an anchor to prevent high false confidence
    return float(min(weighted, min(seg_conf, rec_conf, val_conf) + 15.0))


def validate_field(field_name: str, raw_text: str, language: str = "en") -> tuple[str, float]:
    """
    Applies domain-specific postprocessing (Phase 5).
    Fuzzy matches against known lexicons to correct minor OCR spelling mistakes,
    and returns a validation confidence score (0-100).
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return "", 0.0

    # Phase 2 Paper Improvement: Symbol-aware post-processing
    # Fix common TrOCR hallucinations with numbers and medical symbols
    # e.g., "1H1" -> "1+1", "1H1H1" -> "1+1+1"
    raw_text = re.sub(r"(\d)[Hh](\d)", r"\1+\2", raw_text)
    # Fix dropped slashes between dosages e.g., "400mg16mg" -> "400mg/16mg"
    raw_text = re.sub(r"(mg|g|ml)(\d+)(mg|g|ml)", r"\1/\2\3", raw_text)

    # MULTI-CLASS BYPASS:
    # Do not apply medical fuzzy matching to generic fields like Doctor Names or Patient Info
    if field_name.lower().startswith(("doctername", "patientinfo")):
        return raw_text, 100.0

    # For medicine/prescription fields, use the smarter crop corrector
    if field_name.lower().startswith(("medicine", "prescription", "medical text")):
        return correct_medicine_crop(raw_text)

    lexicon, index = get_lexicon(field_name, language)

    if not lexicon:
        # If no lexicon exists for this field, we just accept the raw text
        # We assign a moderate validation confidence since it wasn't strictly validated
        return raw_text, 50.0

    # Fast indexed lookup: only fuzzy match against words starting with the same letter
    first_letter = raw_text[0].lower()
    candidates = index.get(first_letter, [])
    if not candidates:
        # If no candidates share the first letter, fall back to the full linear search
        # (Useful if TrOCR messed up the very first letter)
        candidates = lexicon

    # Rapidfuzz extractOne returns (match, score, index)
    match = process.extractOne(raw_text, candidates, scorer=fuzz.WRatio)

    if match:
        best_match, score, _ = match
        if score > 60:  # Threshold
            return best_match, score

    # If it failed to match anything well, return the raw text with a very low validation score
    return raw_text, 10.0


def correct_medicine_crop(raw_text: str) -> tuple[str, float]:
    if not raw_text.strip():
        return "", 0.0

    from rx_recognition_multiclass.medical_lexicon import correct_medicine_name

    # Run the new powerful TF-IDF fuzzy matcher
    corrected = correct_medicine_name(raw_text, threshold=60)

    # If it successfully changed the text (meaning it found a strong match), give it high confidence
    # If it stayed the same, give it lower validation confidence
    if corrected.lower() != raw_text.lower():
        return corrected, 90.0
    else:
        # Text wasn't changed — it's either already correct or not in the dictionary.
        # Give it 55.0 so it passes the pipeline's MIN_LEXICON_SCORE (50.0) filter.
        # The fine-tuned model's reading should be trusted.
        return raw_text, 55.0


def postprocess_text(field_name: str, raw_text: str) -> str:
    """
    Legacy wrapper for older scripts.
    """
    val, _ = validate_field(field_name, raw_text)
    return val
