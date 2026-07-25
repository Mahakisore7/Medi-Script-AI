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

## Next Steps
The local pipeline structure is complete! The next phase is to deploy this to Kaggle/Google Colab and run the fine-tuning scripts on the real Roboflow dataset!
