# Phase 5 — Integration & Structured Record Generation (Weeks 9–11)

**Owner:** whoever's free — this is the phase where Pods B/C/D's separate modules become one thing. Good to have one person from each pod present for integration, since interface mismatches surface here.

## 5.1 The unified pipeline package

```python
# ml/pipeline/prescription_pipeline.py
from dataclasses import dataclass, field
from segmentation.detector import FieldDetector
from recognition.model import Recognizer
from postprocess.correction import correct_against_lexicon, fix_dosage_symbols
from postprocess.multilingual import map_to_valid_word

@dataclass
class FieldResult:
    field_type: str
    text: str
    confidence: float

@dataclass
class PrescriptionRecord:
    patient_name: FieldResult
    age: FieldResult
    gender: FieldResult
    symptoms: FieldResult
    medicine_and_dosage: FieldResult

class PrescriptionPipeline:
    def __init__(self, detector: FieldDetector, recognizer: Recognizer, lexicons: dict):
        self.detector = detector
        self.recognizer = recognizer
        self.lexicons = lexicons

    def run(self, image) -> PrescriptionRecord:
        crops = self.detector.detect(image)
        results = {}
        for crop in crops:
            raw_text = self.recognizer.transcribe(crop.image)
            if crop.field_type == "medicine_and_dosage":
                raw_text = fix_dosage_symbols(raw_text)
                text, lex_score = correct_against_lexicon(raw_text, self.lexicons["medicine"])
            elif crop.field_type == "symptoms":
                text, lex_score = correct_against_lexicon(raw_text, self.lexicons["symptom"])
            else:
                text, lex_score = raw_text, 1.0
            confidence = 0.5 * crop.confidence + 0.5 * lex_score
            results[crop.field_type] = FieldResult(crop.field_type, text, confidence)
        return PrescriptionRecord(**results)
```
This is the single object every other layer (the API, the evaluation harness, a CLI demo) calls into — one entry point, testable in isolation with `pytest` before the API even exists.

## 5.2 Structured output schema (Pydantic — shared with the API in Phase 6)

```python
from pydantic import BaseModel

class FieldOutput(BaseModel):
    value: str
    confidence: float
    needs_review: bool

class StructuredPrescription(BaseModel):
    patient_name: FieldOutput
    age: FieldOutput
    gender: FieldOutput
    symptoms: FieldOutput
    medicine_and_dosage: FieldOutput

    def to_json(self) -> str: ...
    def to_csv_row(self) -> dict: ...
```
`needs_review = confidence < threshold` (threshold tuned in Phase 4). This schema is what gets exported to JSON/CSV for the EHR/pharmacy-integration deliverable from the synopsis.

## 5.3 Integration tests

```python
def test_pipeline_end_to_end(sample_prescription_image):
    result = pipeline.run(sample_prescription_image)
    assert result.medicine_and_dosage.text != ""
    assert 0 <= result.medicine_and_dosage.confidence <= 1
```
Run this in CI against a handful of committed sample images (small, a few KB each — fine to keep in git unlike the full dataset) so a broken interface between modules fails the build instead of surfacing during a live demo.

## Definition of done for Phase 5

- [ ] `PrescriptionPipeline.run(image)` works end-to-end on a real photo
- [ ] Pydantic schema finalized and shared with Pod E (Product) for the API contract
- [ ] JSON and CSV export both implemented and tested
- [ ] A handful of integration tests running in CI
