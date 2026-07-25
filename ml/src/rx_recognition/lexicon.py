import csv
from collections import defaultdict
from pathlib import Path


def _load_lexicon(path: Path) -> list[str]:
    """Loads a simple single-column CSV into a list of strings."""
    if not path.exists():
        # Fallback for when the CSVs haven't been generated yet
        return []
        
    with open(path, encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f) if row]


def _build_index(lexicon: list[str]) -> dict[str, list[str]]:
    """
    Builds a first-letter index to dramatically speed up fuzzy matching.
    Instead of O(N) over thousands of entries, we only check words that 
    share the same first letter as our guess.
    """
    index = defaultdict(list)
    for entry in lexicon:
        if entry:
            first_letter = entry[0].lower()
            index[first_letter].append(entry)
    return index


# Load data from the data/lexicon directory relative to this script
# (ml/src/rx_recognition/lexicon.py -> ../../../../data/lexicon/)
_lexicon_dir = Path(__file__).parent.parent.parent.parent / "data" / "lexicon"

MEDICINE_LEXICON = _load_lexicon(_lexicon_dir / "medicines.csv")
SYMPTOM_LEXICON = _load_lexicon(_lexicon_dir / "symptoms.csv")

MEDICINE_INDEX = _build_index(MEDICINE_LEXICON)
SYMPTOM_INDEX = _build_index(SYMPTOM_LEXICON)


def get_lexicon(field_name: str, language: str = "en") -> tuple[list[str], dict[str, list[str]]]:
    """
    Returns the appropriate lexicon list and its first-letter index for a given field.
    Prepared for the multilingual stretch goal.
    """
    if field_name == "medicine":
        return MEDICINE_LEXICON, MEDICINE_INDEX
    elif field_name == "symptoms":
        return SYMPTOM_LEXICON, SYMPTOM_INDEX
    return [], {}
