# Medi-Script-AI: Pure DL Handwriting Recognition Pipeline

Medi-Script-AI is an end-to-end deep learning pipeline for automatically segmenting and recognizing messy handwritten medical prescriptions.

Because medical prescriptions rely heavily on domain-specific vocabulary (complex drug names and dosages) and feature highly cursive, connected, and degraded handwriting, traditional OCR engines (like Tesseract or EasyOCR) struggle immensely. To solve this, this repository implements a custom hybrid architecture combining object detection (YOLOv8) and transformer-based handwriting recognition (TrOCR).

## Architecture Overview

The pipeline consists of four main phases:

1. **YOLOv8 Segmentation:**
   We trained a custom YOLOv8 model on the **Bannari-Amman Kaggle Dataset**. We specifically annotated and extracted only the `medicine` bounding boxes to create a specialized Roboflow dataset. This allows the pipeline to ignore noise (logos, clinic names, doctor signatures) and strictly isolate individual medicine lines.

2. **Crop Enhancement:**
   Raw YOLO crops are often low-contrast, inconsistently lit, or too thin. The `crop_enhancer.py` applies adaptive thresholding (CLAHE), grayscale conversion, and morphological dilation to thicken the pen strokes and maximize contrast before OCR.

3. **TrOCR-Large (Transformer-based OCR):**
   Instead of traditional CNN-RNN architectures, we use Microsoft's **TrOCR-Large** (558 million parameters). By processing the enhanced crop through a Vision Transformer (ViT) encoder and a RoBERTa language decoder, TrOCR leverages its pre-training on massive handwriting datasets (IAM, SROIE) to decode the cursive strokes visually.

4. **Robust Lexicon Matching:**
   Because handwriting is inherently ambiguous (e.g., `Ecosprin` vs `Ecoseptic`), we process the raw output through a domain-specific drug lexicon (containing 250,000+ Indian medicines). The `postprocess.py` module uses Levenshtein distance combined with a prefix-weighted bonus to snap phonetic misspellings back to the correct medical terms.

## Repository Structure

- `demo.py`: The main command-line interface for running inference on new prescription images.
- `ml/src/rx_recognition/`: The core pipeline source code.
  - `pipeline.py`: Orchestrates segmentation, enhancement, OCR, and post-processing.
  - `recognition.py`: Handles loading and running the TrOCR HuggingFace models.
  - `segmentation.py`: Handles YOLO bounding box extraction.
  - `crop_enhancer.py`: OpenCV image processing for crop contrast/dilation.
  - `postprocess.py`: Handles the robust fuzzy string matching algorithm against the medical lexicon.
  - `generate_trocr_data.py`: Script to generate synthetic cursive training data.

## Getting Started

### 1. Installation
Ensure you have `uv` installed, then synchronize the environment:
```bash
uv sync
```

### 2. Download the Models
- **YOLOv8 Weights:** Place your fine-tuned `yolo_real_world_best.pt` inside `runs/detect/`.
- **TrOCR Base:** The pipeline automatically downloads `microsoft/trocr-large-handwritten` from Hugging Face on first run.

### 3. Run Inference
You can test the pipeline on a raw prescription image by running:
```bash
uv run demo.py "path/to/prescription.jpg" --mode yolo
```
The pipeline will output a structured dictionary containing the exact medicine names and dosages detected, and it will save a debug image (`runs/debug/segmentation_debug.jpg`) showing the extracted YOLO bounding boxes.

## Advanced: Fine-Tuning TrOCR on Custom Data

If you wish to fine-tune TrOCR on local pharmacy data, you can generate a synthetic dataset and run the training scripts:

1. **Generate Synthetic Data:**
   Use the `generate_trocr_data.py` script to generate thousands of synthetic images using different cursive fonts and random Indian medicine combinations.
2. **Train on Colab/Kaggle:**
   Due to the massive size of TrOCR-Large, you must run `train_trocr.py` on a GPU instance (T4 or higher). Ensure you use `per_device_train_batch_size=2` and `gradient_checkpointing=True` to avoid Out-Of-Memory errors on free tiers.
