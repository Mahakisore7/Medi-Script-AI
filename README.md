# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## Branch: `feat/baseline-eval` (Phase 2)

This branch focuses on **Phase 2: Baseline Walking Skeleton**. We are building out the basic, end-to-end pipeline using placeholders (OpenCV and Tesseract) to establish a baseline Character Error Rate (CER) and Word Error Rate (WER).

### Key Components
- `ml/src/rx_recognition/preprocessing.py`: Basic OpenCV processing.
- `ml/src/rx_recognition/segmentation.py`: Contour-based placeholder.
- `ml/src/rx_recognition/recognition.py`: Tesseract baseline.
- `ml/src/rx_recognition/evaluation.py`: Computes CER, WER, EMA using RapidFuzz.

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
