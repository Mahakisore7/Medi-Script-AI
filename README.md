# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## Branch: `feat/trocr-finetune` (Phase 4)

This branch focuses on **Phase 4: Text Recognition Engine**. We replaced the initial Tesseract OCR placeholder with a state-of-the-art Hugging Face `TrOCR` (VisionEncoderDecoder) model.

### Key Components
- `ml/src/rx_recognition/train_trocr.py`: Script to fine-tune the `microsoft/trocr-base-handwritten` model on custom prescription data.
- `ml/src/rx_recognition/generate_trocr_data.py`: Script to generate training manifests from Roboflow or synthetic datasets.
- `ml/src/rx_recognition/recognition.py`: Implements `TrOCRRecognizer` for pipeline inference.
- `KAGGLE_TRAINING_NOTES.md`: Notes for running the actual training in the cloud.

### Local Setup

We use `uv` for reproducible, fast Python environment management.

1. **Install uv**:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```
2. **Install dependencies**:
   ```bash
   cd ml
   uv sync
   ```
3. **Run the baseline evaluation**:
   ```bash
   uv run --project ml python -m rx_recognition.run_baseline_eval
   ```

*(Note: The old IAM pretraining proof-of-concept can be found in the `iam_pretraining_poc/` directory).*
