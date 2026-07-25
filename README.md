# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## Branch: `feat/segmentation-yolo` (Phase 3)

This branch focuses on **Phase 3: Field Segmentation**. We replaced the brittle OpenCV horizontal projection chunker with a robust YOLOv8 Object Detection model.

### Key Components
- `ml/src/rx_recognition/train_yolo.py`: Training script for fine-tuning a YOLOv8 object detector to crop prescription fields.
- `ml/src/rx_recognition/generate_yolo_data.py`: Data conversion script for synthetic testing.
- `ml/src/rx_recognition/segmentation.py`: Implements `TrainedFieldDetector` using `ultralytics` to seamlessly integrate YOLO inference into the pipeline.

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
