from rapidfuzz import fuzz

from rx_recognition_multiclass.lexicon_trie import MedicalLexicons


def reconcile_predictions(trocr_text: str, easyocr_text: str, field_type: str) -> tuple[str, str]:
    """
    Cross-checks TrOCR against EasyOCR.
    Returns: (final_text, confidence_level)
    confidence_level is one of: "High", "Medium", "Low"
    """
    trocr_clean = trocr_text.strip().lower()
    easyocr_clean = easyocr_text.strip().lower()

    if not trocr_clean and not easyocr_clean:
        return "", "Low"

    if not trocr_clean:
        return easyocr_text, "Low"

    if not easyocr_clean:
        return trocr_text, "Low"

    # Check for agreement (fuzzy match > 85%)
    agreement = fuzz.ratio(trocr_clean, easyocr_clean) > 85

    if agreement:
        return trocr_text, "High"

    # Disagreement. Check lexicon.
    lexicons = MedicalLexicons()

    trocr_in_lex = False
    easyocr_in_lex = False

    if field_type == "medicine":
        trie = lexicons.get_medicines_trie()
        trocr_in_lex = trie.is_valid_word(trocr_clean)
        easyocr_in_lex = trie.is_valid_word(easyocr_clean)
    elif field_type == "symptoms":
        trie = lexicons.get_symptoms_trie()
        trocr_in_lex = trie.is_valid_word(trocr_clean)
        easyocr_in_lex = trie.is_valid_word(easyocr_clean)

    if trocr_in_lex and easyocr_in_lex:
        # Both in lexicon but disagree? Pick TrOCR but medium confidence
        return trocr_text, "Medium"
    elif trocr_in_lex:
        return trocr_text, "Medium"
    elif easyocr_in_lex:
        return easyocr_text, "Medium"

    # Neither in lexicon, and they disagree
    return trocr_text, "Low"
