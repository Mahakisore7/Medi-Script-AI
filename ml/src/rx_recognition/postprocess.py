import re

from rapidfuzz import fuzz, process

from rx_recognition.lexicon import get_lexicon


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

    # For medicine/prescription fields, use the smarter crop corrector
    if field_name.lower().startswith(("medicine", "prescription")):
        return correct_medicine_crop(raw_text)

    lexicon, index = get_lexicon(field_name, language)

    if not lexicon:
        # If no lexicon exists for this field, we just accept the raw text
        # We assign a moderate validation confidence since it wasn't strictly validated
        return raw_text, 50.0

    # Fast indexed lookup: only fuzzy match against words starting with the same letter
    first_letter = raw_text[0].lower()
    candidates = index.get(first_letter, [])\

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
    """
    Fix 2: Smarter medicine name correction for YOLO crop OCR output.

    The problem: EasyOCR reads a medicine crop and returns something like:
        "Nabv + 1 0 X2"
    which contains BOTH the medicine name ("Nabv") AND the dosage ("1 0 X2").

    This function:
      1. Splits the raw text into candidate name tokens (first 1-3 words before digits)
      2. Fuzzy-matches each candidate against the 1142-drug lexicon
      3. Picks the best matching drug name
      4. Preserves the dosage tail as-is
      5. Returns "CorrectedDrugName DosageTail"
    """
    if not raw_text.strip():
        return "", 0.0

    lexicon, index = get_lexicon("medicine")
    if not lexicon:
        return raw_text, 10.0

    tokens = raw_text.strip().split()

    # Find where the dosage starts — first token that is purely numeric or looks like a dose
    # e.g. "1", "0", "X2", "40mg", "1+0+1", "BD", "OD", "TDS"
    DOSAGE_TOKENS = re.compile(
        r"^(\d[\d.]*|x\d+|\d+mg|\d+ml|\d+g|bd|od|tds|hs|stat|qid|prn|sos|\+|\-|x|/|tab|cap|syr)$",
        re.IGNORECASE
    )

    name_tokens = []
    dosage_tokens = []
    found_dosage = False
    for tok in tokens:
        if not found_dosage and not DOSAGE_TOKENS.match(tok) and not tok.isdigit():
            name_tokens.append(tok)
        else:
            found_dosage = True
            dosage_tokens.append(tok)

    # Try progressively shorter candidate phrases: trigram → bigram → unigram
    best_drug = None
    best_score = 0.0
    for n in range(min(3, len(name_tokens)), 0, -1):
        candidate_phrase = " ".join(name_tokens[:n])

        if len(candidate_phrase) < 3:
            continue

        first_letter = candidate_phrase[0].lower() if candidate_phrase else ""
        candidates = index.get(first_letter, lexicon)

        # Robust Custom Scoring:
        # 1. Base Levenshtein ratio (requires lengths to be similar, unlike WRatio)
        # 2. Huge bonus if the first 3-4 letters match perfectly (doctors write prefixes clearly)
        # 3. Penalty if the length difference is too large (prevents "Bisoder" matching "Sanipred")
        
        cand_lower = candidate_phrase.lower()
        for drug in candidates:
            drug_lower = drug.lower()
            
            # Length penalty
            len_diff = abs(len(cand_lower) - len(drug_lower))
            if len_diff > 5:
                continue  # Skip wildly different lengths
                
            base_score = fuzz.ratio(cand_lower, drug_lower)
            
            # Prefix bonus
            prefix_match = 0
            for c1, c2 in zip(cand_lower, drug_lower):
                if c1 == c2:
                    prefix_match += 1
                else:
                    break
                    
            if prefix_match >= 3:
                base_score += 15
            elif prefix_match >= 2:
                base_score += 5
                
            final_score = min(100.0, base_score)
            
            if final_score > best_score:
                best_score = final_score
                best_drug = drug

    # Only accept the lexicon correction if confidence is high enough.
    # Threshold of 65 avoids wrong matches (e.g. 'bozx' → random drug).
    ACCEPTANCE_THRESHOLD = 65
    if best_drug and best_score >= ACCEPTANCE_THRESHOLD:
        dosage_tail = " ".join(dosage_tokens).strip()
        final = f"{best_drug} {dosage_tail}".strip()
        return final, best_score
    else:
        # Not confident enough — return raw text to avoid wrong corrections
        return raw_text, 10.0


def postprocess_text(field_name: str, raw_text: str) -> str:
    """
    Legacy wrapper for older scripts.
    """
    val, _ = validate_field(field_name, raw_text)
    return val
