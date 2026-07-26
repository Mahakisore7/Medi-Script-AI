# Medi-Script-AI: Technical Project Summary

This document outlines the entire technical lifecycle of the Medi-Script-AI project. It breaks down the architecture into distinct phases, detailing the exact models used, the datasets trained on, the motivation behind each step, and the final outcome achieved.

---

## 1. Architecture Design & Environment Setup
**Motive:**
Medical prescriptions combine unstructured layouts (variable hospital paper formats) with highly irregular cursive handwriting. A monolithic AI model cannot solve this effectively. We designed a **Multi-Stage Pipeline**: Segmentation $\rightarrow$ OCR $\rightarrow$ Lexical Post-processing.
**Tools Used:**
- **Language/Environment**: Python 3.12 managed via `uv` package manager for ultra-fast dependency resolution.
- **Key Libraries**: `torch`, `ultralytics` (for YOLO), `transformers` (for HuggingFace models), `opencv-python` (for image manipulation), and `rapidfuzz` (for string matching).

---

## 2. Phase 1: Region of Interest (ROI) Segmentation
**Motive:**
Feeding an entire prescription image (with hospital logos, printed lines, and blank whitespace) directly into a text-reading AI causes severe hallucination. The motive of this phase was to build an object detector that ignores printed noise and draws tight bounding boxes strictly around handwritten ink.
**Technical Details:**
- **Model Used:** **YOLOv8n** (You Only Look Once, version 8 nano). We chose the 'nano' variant for extremely fast inference speeds while maintaining high accuracy.
- **Dataset Used:** We utilized a public Roboflow dataset named **"Medical Prescription Annotation"** which contained roughly 500+ images of prescriptions. The dataset was labeled with a single class: `handwritten`.
- **Outcome:** We built the `segmentation.py` module. It takes a raw image matrix, runs YOLO inference, filters boxes by a confidence threshold (tuned down to `0.01` to capture faint handwriting), and returns cropped NumPy arrays of the detected text regions.

---

## 3. Phase 2: Handwriting Recognition (OCR)
**Motive:**
Once we had perfectly cropped images of handwriting, we needed an AI capable of reading cursive English. Standard OCR engines (like PyTesseract) are built for printed text and fail completely on cursive.
**Technical Details:**
- **Model Used:** Microsoft's **TrOCR (Transformer-based Optical Character Recognition)**, specifically the `microsoft/trocr-base-handwritten` checkpoint from HuggingFace.
- **Architecture:** TrOCR bypasses traditional CNNs entirely. It uses a **Vision-Encoder-Decoder** architecture (a Vision Transformer (ViT) for image encoding, and an NLP RoBERTa-style decoder for text generation). It was pre-trained on the massive IAM Handwriting dataset.
- **Outcome:** We built `recognition.py`. It takes the small cropped images from YOLO, normalizes them, passes them through the Vision Transformer, and decodes the pixel features into raw string text.

---

## 4. Phase 3: Lexicon Matching & Post-Processing
**Motive:**
Even TrOCR will occasionally misread messy doctor handwriting (e.g., reading "Amoxicillin" as "Amoxcilin"). In the medical domain, spelling mistakes are unacceptable. This phase acts as a safety layer to map raw AI predictions to valid medical terminology.
**Technical Details:**
- **Algorithm Used:** **Fuzzy String Matching** using the Levenshtein distance algorithm (via the highly optimized `rapidfuzz` library).
- **Dataset Used:** A custom JSON Lexicon (`fda_drugs.json`) containing standardized drug names based on real FDA databases, as well as common symptoms and dosage instructions.
- **Outcome:** We built `postprocess.py`. It calculates the string similarity ratio between the TrOCR output and the lexicon. If the ratio exceeds a threshold (e.g., 60%), it snaps the text to the closest valid medical term and outputs a final, combined confidence score.

---

## 5. Phase 4: Cloud Training and Fine-Tuning
**Motive:**
While our architecture was flawless, YOLOv8 out-of-the-box is trained on the COCO dataset (common objects like cars and dogs). It cannot detect medical handwriting zero-shot. Training Object Detection models requires GPU compute, which is difficult to do locally.
**Technical Details:**
- **Platform:** **Kaggle** (utilizing 2x NVIDIA Tesla T4 GPUs).
- **Process:** We uploaded our Roboflow dataset to Kaggle and fine-tuned the base `yolov8n.pt` weights. We trained the model for **50 epochs** with an image size of `640x640`.
- **Outcome:** The model achieved an impressive **0.887 mAP50** (Mean Average Precision). We downloaded these new, custom weights (`kaggle_best.pt`) to our local machine.

---

## 6. Phase 5: Pipeline Integration
**Motive:**
To create a seamless, end-to-end product that a user can interact with easily.
**Technical Details:**
- **Implementation:** We built `pipeline.py` and `demo.py`. This orchestrates the flow of data:
  1. Loads image via OpenCV.
  2. Passes image to the custom-trained YOLO model for bounding boxes.
  3. Loops through all crops and passes them to the TrOCR pipeline.
  4. Runs the output through the fuzzy matcher.
  5. Compiles everything into a structured JSON dictionary.
- **Outcome:** A production-ready, modular Deep Learning pipeline that successfully detects, crops, reads, and corrects real-world medical handwriting.
