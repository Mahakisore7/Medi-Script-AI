# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## Branch: `feat/data-strategy` (Phase 1)

This branch focuses on **Phase 1: Data Strategy**. We are establishing a storage-efficient, ethically-sound data pipeline.
Rather than writing augmented images to disk, we use in-memory augmentations via Albumentations.

### Key Components
- `data/DATA_MANIFEST.md`: Ledger tracking our external datasets (e.g., Roboflow).
- `data/DATASET_CARD.md`: Documentation on the 3-way dataset split and biases.
- `ml/src/rx_recognition/augment.py`: In-memory data augmentation pipeline.
- `ml/pyproject.toml`: Project dependencies managed by `uv`.

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
3. **Run the augmentation tests**:
   ```bash
   uv run --project ml pytest ml/tests -v
   ```

*(Note: The old IAM pretraining proof-of-concept can be found in the `iam_pretraining_poc/` directory).*
