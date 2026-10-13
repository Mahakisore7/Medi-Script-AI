# Lexicons

The pipeline expects two single-column CSV files in this directory:

| File | Content | Distribution |
|---|---|---|
| `medicines.csv` | 384,604 Indian medicine brand names | Not committed (too large / licence); build with `python scripts/fetch_lexicon.py` |
| `symptoms.csv` | 35 common symptom phrases | Small, may be committed |

Without `medicines.csv` the fuzzy lexicon-correction stage is skipped and recognised text is returned as read.
