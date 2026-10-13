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
from mediscript.config import LEXICON_DIR as _lexicon_dir

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


def correct_medicine_name(text, threshold=75):
    """
    Given a noisy medicine name from OCR, tries to correct it against the
    massive medicine lexicon. Uses TF-IDF for instant lookup across 380k entries.
    """
    clean_text = text.strip(" .,;:-()")
    if not clean_text or len(clean_text) < 3:
        return text

    # Extract the potential medicine name by skipping leading dosages/numbers
    tokens = clean_text.split()

    dosage_keywords = {
        "tab",
        "cap",
        "syr",
        "syp",
        "syrup",
        "inj",
        "iv",
        "oral",
        "suspension",
        "tablet",
        "capsule",
        "drop",
        "drops",
        "ointment",
        "cream",
        "bd",
        "od",
        "tds",
        "sos",
        "hs",
        "ac",
        "pc",
        "mg",
        "ml",
        "gm",
        "per",
        "mcg",
    }

    # Find candidate sequences of words (skipping numbers and dosage keywords)
    tokens = clean_text.split()
    candidate_sequences = []
    current_seq = []

    for token in tokens:
        clean_tok = re.sub(r"[^\w]", "", token)
        if not clean_tok:
            continue
        if re.search(r"\d", clean_tok) or clean_tok.lower() in dosage_keywords:
            if current_seq:
                candidate_sequences.append(current_seq)
                current_seq = []
        else:
            if len(clean_tok) > 1:  # Ignore single letters like 'f' or 'x'
                current_seq.append(token)  # Keep original token for replacement

    if current_seq:
        candidate_sequences.append(current_seq)

    if not candidate_sequences:
        return text

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

    replacements = []

    for seq in candidate_sequences:
        best_seq_score = 0
        best_seq_match = None
        best_seq_tokens = None

        for n in range(len(seq), 0, -1):
            candidate = " ".join(seq[:n])
            if len(candidate) < 3:
                continue

            score, match = _find_best_match(candidate)
            if score >= threshold and score > best_seq_score:
                best_seq_score = score
                best_seq_match = match
                best_seq_tokens = seq[:n]

        if best_seq_match and best_seq_tokens:
            replacements.append((best_seq_tokens, best_seq_match))

    for tokens_to_replace, match in replacements:
        pattern = r"\s+".join([re.escape(t) for t in tokens_to_replace])
        text = re.sub(pattern, match, text, count=1, flags=re.IGNORECASE)

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
    # Only touch tokens that look like dosage notation (digits plus OCR-confusable
    # glyphs); never rewrite letters inside drug names such as "Hall" or "Ozad".
    def _fix(m):
        return m.group(0).replace("H", "+").replace("l", "1").replace("O", "0")

    return re.sub(r"(?<![A-Za-z])(?=[0-9HlO+\-/]*[0-9])[0-9HlO+\-/]{3,}(?![A-Za-z])", _fix, text)
