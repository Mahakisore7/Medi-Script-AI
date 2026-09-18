import csv
import re
from pathlib import Path

from rapidfuzz import fuzz

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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


# Resolve paths relative to this file:
_this_dir = Path(__file__).parent
_lexicon_dir = _this_dir.parent.parent.parent / "data" / "lexicon"

MEDICINE_LIST = _load_csv(_lexicon_dir / "medicines.csv")
SYMPTOM_LIST = _load_csv(_lexicon_dir / "symptoms.csv")

print(f"[medical_lexicon] Loaded {len(MEDICINE_LIST)} medicines, {len(SYMPTOM_LIST)} symptoms")

if HAS_SKLEARN and MEDICINE_LIST:
    print("[medical_lexicon] Building TF-IDF index for fast fuzzy matching...")
    # Use character n-grams from length 2 to 4 to handle spelling typos
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(MEDICINE_LIST)
else:
    print("[medical_lexicon] scikit-learn not found or empty lexicon. Falling back to slow search.")
    vectorizer = None
    tfidf_matrix = None


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


def correct_medicine_name(text, threshold=60):
    """
    Given a noisy medicine name from OCR, tries to correct it against the
    massive medicine lexicon. Uses TF-IDF for instant lookup across 380k entries,
    and strips dosage prefixes to avoid punishing exact core matches.
    """
    clean_text = text.strip(" .,;:-()")
    if not clean_text or len(clean_text) < 3:
        return text

    # Pre-filtering: Remove common dosage words to find the core medicine name
    # e.g., "Tab Paracetamol 500mg" -> "Paracetamol"
    core_text = re.sub(
        r"\b(tab|cap|syr|inj|iv|oral|suspension|tablet|capsule|drop|drops|ointment|cream)\b",
        "",
        clean_text,
        flags=re.IGNORECASE,
    )
    core_text = re.sub(r"\b\d+(\.\d+)?\s*(mg|ml|gm|mcg|g)\b", "", core_text, flags=re.IGNORECASE)
    core_text = re.sub(r"\b\d+\b", "", core_text)  # Remove standalone numbers only
    core_text = core_text.strip(" .,;:-()")

    if not core_text:
        core_text = clean_text

    def _find_best_match(query: str):
        if HAS_SKLEARN and vectorizer is not None:
            query_vec = vectorizer.transform([query.lower()])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            best_idx = scores.argmax()
            return scores[best_idx] * 100.0, MEDICINE_LIST[best_idx]
        else:
            b_match, b_score = None, 0
            for med in MEDICINE_LIST:
                score = fuzz.token_set_ratio(query.lower(), med.lower())
                if score > b_score:
                    b_score = score
                    b_match = med
            return b_score, b_match

    # Try matching the full core_text first
    tokens = core_text.split()

    # Progressively try shorter phrases (e.g. "Loraladin Compraze" -> "Loraladin")
    # This perfectly handles TrOCR hallucinating extra random words at the end
    best_overall_match = None
    best_overall_score = 0
    best_core_used = core_text

    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n])
        if len(candidate) < 3:
            continue

        score, match = _find_best_match(candidate)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_match = match
            best_core_used = candidate

        if score >= threshold:
            break

    if best_overall_score >= threshold and best_overall_match:
        # Replace the part of the text we matched with the corrected text
        if best_core_used in clean_text:
            return clean_text.replace(best_core_used, best_overall_match)
        return best_overall_match

    return text


def correct_symptom(text, threshold=70):
    """
    Given a noisy symptom from OCR, tries to correct it against the symptom lexicon.
    Uses token_set_ratio to handle extra words.
    """
    clean_text = text.strip(" .,;:-()")
    if not clean_text or len(clean_text) < 3:
        return text

    best_match = None
    best_score = 0

    for sym in SYMPTOM_LIST:
        score = fuzz.token_set_ratio(clean_text.lower(), sym.lower())
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
