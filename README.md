# Medi-Script-AI: Pure DL Handwriting Recognition Pipeline

Medi-Script-AI is an end-to-end deep learning pipeline for automatically segmenting and recognizing messy handwritten medical prescriptions.

Because medical prescriptions rely heavily on domain-specific vocabulary (complex drug names and dosages) and feature highly cursive, connected, and degraded handwriting, traditional OCR engines (like Tesseract or EasyOCR) struggle immensely. To solve this, this repository implements a custom hybrid architecture combining object detection (YOLOv8) and transformer-based handwriting recognition (TrOCR).

## Architecture Overview

The pipeline uses a robust dual-model approach, separating the highly cursive medical prescriptions from standard printed metadata.

1. **Prescription Extraction (YOLOv8 + TrOCR):**
   - **YOLOv8 Segmentation:** We trained a custom YOLOv8 model to specifically isolate `medicine` bounding boxes (individual handwritten prescription lines). This filters out background noise, logos, and printed text.
   - **Crop Enhancement:** Raw YOLO crops undergo adaptive thresholding, grayscale conversion, and morphological dilation to maximize handwriting contrast.
   - **TrOCR-Large:** The enhanced crops are passed to Microsoft's TrOCR-Large model, which excels at decoding complex cursive medical handwriting.
   - **Robust Lexicon Matching:** The raw TrOCR output is fuzzy-matched against a vast Indian medical lexicon (250,000+ drugs) to automatically correct spelling errors and resolve ambiguities.

2. **Metadata Extraction (Global OCR + Spatial Parsing):**
   - **Global Text Detection:** While YOLO handles the medicines, an independent OCR engine scans the entire document for printed text such as clinic headers, doctor names, contact information, and dates.
   - **Spatial Heuristics:** The system analyzes the spatial coordinates (bounding boxes) of the detected text to map related fields. For example, by finding anchor words like "Name", "Age", or "Sex", the parser searches nearby spatial regions to extract the corresponding patient values.
   - **Targeted Fallbacks:** If the global OCR misses challenging or single-character handwritten entries (like "M" or "F" for Sex), the system falls back to cropping the isolated spatial region and analyzing it directly with TrOCR.

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
uv run demo.py "path/to/prescription.jpg" --mode heuristic
```
The pipeline will output a structured JSON dictionary containing the extracted Patient Information (Name, Age, Sex), Doctor and Hospital Metadata, and the exact medicine names and dosages detected.

## Advanced: Fine-Tuning TrOCR on Custom Data

If you wish to fine-tune TrOCR on local pharmacy data, you can generate a synthetic dataset and run the training scripts:

1. **Generate Synthetic Data:**
   Use the `generate_trocr_data.py` script to generate thousands of synthetic images using different cursive fonts and random Indian medicine combinations.
2. **Train on Colab/Kaggle:**
   Due to the massive size of TrOCR-Large, you must run `train_trocr.py` on a GPU instance (T4 or higher). Ensure you use `per_device_train_batch_size=2` and `gradient_checkpointing=True` to avoid Out-Of-Memory errors on free tiers.
