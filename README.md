# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, building an end-to-end pipeline for handwritten medical prescription recognition.

## The Core Architecture

This pipeline chains three distinct AI modules together. A raw image goes in one end, and structured JSON data comes out the other:

1. **YOLOv8 (Segmentation Module):** Acts as the "eyes". Instead of reading the whole page at once, YOLO scans the document and draws precise bounding boxes around specific fields (e.g., Patient Name, Medicine, Age).
2. **TrOCR (Recognition Module):** Acts as the "reader". It takes the small cropped image of handwriting and uses a VisionEncoderDecoder Transformer to translate the pixels into raw text characters.
3. **Lexicon & RapidFuzz (Post-processing Module):** Acts as the "spellchecker". Because handwriting AI often makes slight spelling mistakes, this module uses fuzzy-string matching to compare TrOCR's raw output against a known dictionary of FDA medical terms to autocorrect it.

---

## 🚀 Quick Start Guide (Reproduce the Project)

Follow these steps to generate the data, train the model, and run the pipeline from scratch.

### Step 1: Environment Setup
We use `uv` for reproducible, fast Python environment management.
1. Install uv:
   ```powershell
   irm https://astral.sh/uv/install.ps1 | iex
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```

### Step 2: Generate the Synthetic Dataset
Instead of manually annotating hundreds of prescriptions, we wrote a procedural Python script to generate synthetic medical prescriptions with automatic YOLO bounding boxes.
```bash
python dataset_generation/generate_prescription_dataset.py
```
This will generate a `prescription_dataset` folder locally. Zip this folder into `synthetic_prescriptions.zip` (Ensure you use a tool that zips with Linux forward-slashes for Kaggle compatibility!).

### Step 3: Train the YOLO Model (Kaggle)
1. Go to Kaggle and create a new Notebook.
2. Click **Add Input -> New Dataset** and upload your `synthetic_prescriptions.zip` file.
3. Upload the `kaggle_training.ipynb` file from this repository into Kaggle and run the cells to train YOLOv8 on your dataset for 50 epochs.
4. Download the resulting `best.pt` weights file, rename it to `kaggle_best.pt`, and place it inside your local `runs/detect/` folder.

### Step 4: Run the Inference Pipeline
We have included a synthetic sample image in the repo for you to test out of the box. Run the demo script:
```bash
uv run --project ml python demo.py
```
This will run the full AI pipeline and output the `Final Structured Record` as a JSON dictionary!

---

## Technical Architecture & Results
For a complete, highly-detailed breakdown of the exact Deep Learning models used, and the final metrics achieved (0.994 mAP50), please read the **[`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md)** file!
