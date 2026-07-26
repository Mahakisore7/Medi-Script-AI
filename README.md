# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## Local Setup

We use `uv` for reproducible, fast Python environment management. To get started:

1. **Install uv**:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```
2. **Install dependencies**:
   ```bash
   cd ml
   uv sync
   ```

*(Note: The old IAM pretraining proof-of-concept can be found in the `iam_pretraining_poc/` directory).*

---

## Project Phases & Progress

### Phase 1: Data Strategy
**Branch:** `feat/data-strategy`
- Established the data collection strategy using synthetic images for local testing.
- Created dataset cards and manifest protocols for Kaggle storage.
- Implemented data augmentation logic using `albumentations` for on-the-fly transformations.
> **How to verify:**
> Run the automated tests to verify the augmentation pipeline functions correctly in-memory:
> ```bash
> uv run --project ml pytest ml/tests -v
> ```

### Phase 2: Baseline Pipeline
**Branch:** `feat/baseline-eval`
- Constructed a baseline "Walking Skeleton" of the pipeline using simple placeholders.
- **Segmentation:** OpenCV contour-based horizontal projection.
- **Recognition:** Tesseract OCR.
- Established an automated evaluation harness with RapidFuzz for calculating Character Error Rate (CER), Word Error Rate (WER), and Exact Match Accuracy (EMA).
> **How to verify:**
> Run the full baseline pipeline evaluation (this will output your baseline CER/WER):
> ```bash
> uv run --project ml python -m rx_recognition.run_baseline_eval
> ```

### Phase 3: Field Segmentation (YOLOv8)
**Branch:** `feat/segmentation-yolo`
- Replaced the brittle OpenCV chunker with a robust **YOLOv8 Object Detection** model.
- Integrated `ultralytics` for field extraction.
- Developed a local smoke-test script to train YOLO on synthetic data, proving out the workflow for future cloud-based fine-tuning on real datasets.
- Connected the `TrainedFieldDetector` into the pipeline.
> **How to verify:**
> First, generate the synthetic dataset (YOLO format), then run a quick 1-epoch smoke test:
> ```bash
> uv run --project ml python -m rx_recognition.generate_yolo_data
> uv run --project ml python -m rx_recognition.train_yolo
> ```

### Phase 4: Text Recognition Engine (TrOCR)
**Branch:** `feat/trocr-finetune`
- Replaced Tesseract with **Microsoft's TrOCR** (`microsoft/trocr-base-handwritten`).
- Created a robust Hugging Face `Seq2SeqTrainer` script (`train_trocr.py`) for fine-tuning.
- Fixed complex Windows-specific Tokenizer bugs by manually instantiating `ViTImageProcessor` and `RobertaTokenizer`.
- Ran local smoke tests on dummy datasets to confirm the architecture can train (using batch sizes customized for minimal local VRAM).
- Integrated `TrOCRRecognizer` into the pipeline for end-to-end inference!
> **How to verify:**
> Generate the TrOCR manifest, run the local smoke test (2 steps), and evaluate the pipeline:
> ```bash
> uv run --project ml python -m rx_recognition.generate_trocr_data
> uv run --project ml python -m rx_recognition.train_trocr
> uv run --project ml python -m rx_recognition.run_baseline_eval
> ```

---

## The Core Architecture (How it Works)

This pipeline chains three distinct AI modules together. A raw image goes in one end, and structured JSON data comes out the other:

1. **YOLOv8 (Segmentation Module):**
   - *What it does:* Acts as the "eyes". Instead of reading the whole page at once, YOLO scans the document and draws precise bounding boxes around specific fields (e.g., Patient Name, Medicine, Age).
   - *Why we use it:* It crops the image into small, isolated snippets so the text reader isn't confused by surrounding noise.
2. **TrOCR (Recognition Module):**
   - *What it does:* Acts as the "reader". It takes the small cropped image of handwriting (e.g., the medicine scribble) and uses a VisionEncoderDecoder Transformer to translate the pixels into raw text characters.
3. **Lexicon & RapidFuzz (Post-processing Module):**
   - *What it does:* Acts as the "spellchecker". Because handwriting AI often makes slight spelling mistakes, this module uses fuzzy-string matching to compare TrOCR's raw output against a known dictionary of medical terms to autocorrect it!

---

## Why Kaggle Fine-Tuning is Necessary (The "Human Sample" Test)

Currently, the local codebase uses the **un-trained, base versions** of YOLO and TrOCR. If we run a real human prescription (`human_sample_01.jpeg`) through the `demo.py` script right now, we get this result:

```python
--- Raw OCR Text (Before Lexicon Correction) ---
{'age': 'winda more',
 'gender': "Premiere's life .",
 'medicine': '0 0',
 'name': 'THE FIRST REPORT',
 'symptoms': '0 0'}
```

**Why did it hallucinate "THE FIRST REPORT" and "Premiere's life."?**
- **YOLO** has only been trained on synthetic formats, so it cropped the wrong parts of the human image.
- **TrOCR** has never seen medical cursive, so it frantically tried to find standard English sentences in the messy ink scribbles.
- **The Lexicon** only has 10 mock words in it, so it didn't know how to autocorrect `"0 0"`.

**The Fix:** This is exactly why the next step (Phase 4.5 & Phase 5) is to upload the code to Kaggle, fine-tune the models on 580 real prescriptions to cure the hallucinations, and upgrade the Lexicon to a massive openFDA drug dictionary!

### Phase 5: Post-Processing & Structured Confidence
**Branch:** `main`
- Scaled up the Lexicon system by fetching 1,142+ real drug names from the **openFDA API**.
- Implemented a first-letter fuzzy matching index to dramatically increase autocorrect speed.
- Extracted generation confidences from TrOCR and bounding box confidences from YOLO.
- Implemented a Unified Confidence Calibration system (`postprocess.py`) to flag AI hallucinations with extremely low confidence scores when they fail dictionary validation.
> **How to verify:**
> Run the demo script on a human sample to see the low-confidence flags:
> ```bash
> uv run --project ml python demo.py
> ```

---

## Usage (Running the Pipeline)

To test the pipeline on your own medical prescriptions:
1. Place your prescription image inside the `data/samples/` folder.
2. Update the `image_path` variable in `demo.py` to point to your image.
3. Run the end-to-end demo script from the terminal:
```bash
uv run --project ml python demo.py
```

## Technical Architecture & Results

For a complete, highly-detailed breakdown of the exact Deep Learning models used (YOLOv8n, TrOCR), the datasets trained on (Kaggle/Roboflow), and the final metrics achieved in this project, please read the **[`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md)** file!
