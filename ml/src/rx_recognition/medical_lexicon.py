import csv
from collections import defaultdict
from pathlib import Path

from thefuzz import fuzz

# --- Section header matching ---
MEDICAL_TERMS = {
    "Rx": ["rx", "px", "hx", "tx", "ax", "medicine", "medication", "drugs"],
    "Adv": ["adv", "advice", "advise", "instruction", "instructions"],
    "Symptoms": ["symptoms", "c/o", "complains of", "complaint", "signs", "c/0"],
    "Name": ["name", "patient name", "pt name"],
    "Age": ["age", "yrs", "years", "y/o"],
}


# --- Load real lexicons from CSV ---
def _load_csv(path: Path) -> list[str]:
    """Load a single-column CSV into a list of strings."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def _build_index(entries: list[str]) -> dict[str, list[str]]:
    """Build a first-letter index for fast fuzzy matching."""
    index = defaultdict(list)
    for entry in entries:
        if entry:
            index[entry[0].lower()].append(entry)
    return index


# Resolve paths relative to this file:
# medical_lexicon.py -> ml/src/rx_recognition/ -> ../../.. -> data/lexicon/
_this_dir = Path(__file__).parent
_lexicon_dir = _this_dir.parent.parent.parent / "data" / "lexicon"

MEDICINE_LIST = _load_csv(_lexicon_dir / "medicines.csv")
SYMPTOM_LIST = _load_csv(_lexicon_dir / "symptoms.csv")

MEDICINE_INDEX = _build_index(MEDICINE_LIST)
SYMPTOM_INDEX = _build_index(SYMPTOM_LIST)

print(f"[medical_lexicon] Loaded {len(MEDICINE_LIST)} medicines, {len(SYMPTOM_LIST)} symptoms")


def fuzzy_match_section(text, threshold=70):
    """
    Given a piece of text, determines if it likely represents a section header
    like 'Rx', 'Adv', 'Symptoms', 'Name', or 'Age' using fuzzy matching.
    """
    text_lower = text.strip().lower()
    best_match = None
    best_score = 0

    for section, keywords in MEDICAL_TERMS.items():
        for kw in keywords:
            score = fuzz.ratio(text_lower, kw)
            if score > best_score:
                best_score = score
                best_match = section

    if best_score >= threshold:
        return best_match
    return None


def correct_medicine_name(text, threshold=80):
    """
    Given a noisy medicine name from OCR, tries to correct it against the
    1,142-entry medicine lexicon. Uses first-letter indexing for speed.
    Returns the corrected name if a strong match is found, otherwise returns original.
    """
    clean_text = text.strip(" .,;:-()")
    if not clean_text or len(clean_text) < 3:
        return text

    # Try first-letter indexed lookup first (fast path)
    first_letter = clean_text[0].lower()
    candidates = MEDICINE_INDEX.get(first_letter, [])

    # If no candidates share the first letter, fall back to full list
    # (OCR might have messed up the first character)
    if not candidates:
        candidates = MEDICINE_LIST

    best_match = None
    best_score = 0

    for med in candidates:
        score = fuzz.ratio(clean_text.lower(), med.lower())
        if score > best_score:
            best_score = score
            best_match = med

    if best_score >= threshold:
        return best_match
    return text


def correct_symptom(text, threshold=80):
    """
    Given a noisy symptom from OCR, tries to correct it against the symptom lexicon.
    """
    clean_text = text.strip(" .,;:-()")
    if not clean_text or len(clean_text) < 3:
        return text

    first_letter = clean_text[0].lower()
    candidates = SYMPTOM_INDEX.get(first_letter, [])
    if not candidates:
        candidates = SYMPTOM_LIST

    best_match = None
    best_score = 0

    for sym in candidates:
        score = fuzz.ratio(clean_text.lower(), sym.lower())
        if score > best_score:
            best_score = score
            best_match = sym

    if best_score >= threshold:
        return best_match
    return text


def parse_dosage(text):
    """
    Uses fuzzy logic to detect common dosage patterns like 1+1+1, 1-0-1, etc.
    even if OCR messes up symbols (e.g., 1H+1).
    """
    text = text.replace("H", "+").replace("l", "1").replace("O", "0")
    return text
