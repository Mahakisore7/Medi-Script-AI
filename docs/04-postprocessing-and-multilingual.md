# Phase 4 — Domain-Aware Post-Processing & Multilingual Mapping (Weeks 7–10)

**Owner:** Pod D (NLP/Post-processing). **Goal:** take raw TrOCR output and correct it against real medical knowledge, plus support one additional language beyond English for the MVP.

## 4.1 Why this phase matters more than it looks like it should

Every system in your literature review that added a domain-aware correction layer saw a measurable accuracy improvement — this is the cheapest accuracy gain in the whole project relative to effort, because it doesn't require any more training data or GPU time.

## 4.2 Build a medical lexicon

For the MVP, a curated CSV is enough — you don't need a full drug database integration yet:
```
term,type
Paracetamol,medicine
Ibuprofen,medicine
Amoxicillin,medicine
Fever,symptom
Headache,symptom
...
```
Source terms from public open drug-name lists (e.g., openFDA, WHO INN drug name lists) plus whatever medicine names actually appear in your collected dataset (check the Phase 1 EDA). A few hundred entries covering your dataset's actual vocabulary is far more useful than a huge generic list that doesn't match what your volunteers wrote.

## 4.3 Fuzzy/Levenshtein correction

```python
from rapidfuzz import process, fuzz

def correct_against_lexicon(predicted: str, lexicon: list[str], threshold: int = 80) -> tuple[str, float]:
    match, score, _ = process.extractOne(predicted, lexicon, scorer=fuzz.WRatio)
    if score >= threshold:
        return match, score / 100
    return predicted, 0.0   # low confidence — leave as-is and flag for human review
```
Apply this only to `medicine` and `symptoms` field types — don't fuzzy-match `patient_name` against a medical lexicon, that will actively make names worse.

## 4.4 Symbol-aware correction for dosage

Targeted rules for the specific confusions your literature review and your own Phase 3 confusion analysis surfaced:
```python
import re

DOSAGE_CORRECTIONS = [
    (re.compile(r'(?<=\d)H(?=\+|\d)'), '+'),   # "1H+1" -> "1+1" (H misread for +)
    (re.compile(r'(?<=\d{2,3}mg)1(?=\d{2,3}mg)'), '/'),  # "400mg16mg" -> "400mg/6mg"-style fixes
]

def fix_dosage_symbols(text: str) -> str:
    for pattern, replacement in DOSAGE_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text
```
Build these rules from **your own model's actual errors** (from the Phase 3 confusion matrix), not just copied from the literature — your model may make different specific mistakes than the papers you reviewed.

## 4.5 Multilingual — Unicode mapping layer

For the MVP, support English plus **one** regional language your team has native/fluent speakers for (pick whichever your volunteer pool actually wrote prescriptions in during Phase 1 — don't add a language you have zero training data for). Approach:

```python
def map_to_valid_word(raw_chars: str, language_lexicon: dict[str, list[str]], lang: str) -> str:
    if raw_chars in language_lexicon[lang]:
        return raw_chars
    corrected, score = correct_against_lexicon(raw_chars, language_lexicon[lang])
    return corrected if score > 0.6 else raw_chars
```
Language detection can be as simple as a Unicode-block check on the raw predicted characters (e.g., is this in the Devanagari/Tamil/Bengali Unicode range?) before routing to the right lexicon — you don't need a full language-ID model for two languages.

**Evaluate each language separately** with the same CER/WER metrics — your literature review specifically flagged that no reviewed paper reports rigorous per-language accuracy; doing this properly is a genuine point of differentiation for your report.

## 4.6 Confidence scoring

Combine the recognition model's own token-level confidence (available from TrOCR's `generate(..., output_scores=True)`) with the lexicon-match score from 4.3 into a single per-field confidence used for human-review flagging in the app (Phase 6):
```python
final_confidence = 0.5 * model_confidence + 0.5 * lexicon_match_score
```
Tune the blend and the review threshold empirically against your validation set — don't hardcode 0.5/0.5 as final without checking it against real flagged/unflagged outcomes.

## Definition of done for Phase 4

- [ ] Medical lexicon built from public sources + your own dataset's actual vocabulary
- [ ] Fuzzy correction applied selectively (medicine/symptoms only, not names)
- [ ] Symbol-aware dosage rules built from your model's actual confusion patterns
- [ ] Second language integrated with its own lexicon and evaluated separately
- [ ] Combined confidence score implemented and threshold tuned on validation data
