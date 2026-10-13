# Phase 6: Kaggle Fine-Tuning Guide

This guide walks you through executing the training scripts on Kaggle's free GPUs to teach your AI models how to read real medical prescriptions.

## 1. Setting up the Kaggle Environment

1. Log into [Kaggle](https://www.kaggle.com/) and click **"New Notebook"**.
2. On the right-hand panel, under **Notebook Options**:
   - Set **Accelerator** to **GPU P100** or **GPU T4x2**.
   - Set **Internet** to **On**.

## 2. Setting up the Codebase

Create a new code cell and run the following commands to clone your repository and install the required dependencies (using Kaggle's default `pip`):

```bash
!git clone https://github.com/Mahakisore7/Medi-Script-AI.git
%cd Medi-Script-AI/ml
!pip install -e .
```

## 3. Training YOLOv8 (Segmentation)

YOLOv8 needs to learn how to draw bounding boxes around the 5 specific fields (Name, Age, Gender, Symptoms, Medicine) on a real prescription layout.

### A. Download the Dataset
Create a new code cell and paste the Roboflow snippet you provided. **Make sure to insert your actual API key!**

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("daffodil-international-university-s5vpr").project("doctors-prescriptions-handwriting")
version = project.version(1)
dataset = version.download("yolov8")
```

### B. Run the Training Script
Once downloaded, Roboflow will save the dataset in a folder (usually named the same as the project). You can now run our YOLO training script and point it to the `data.yaml` file inside that downloaded folder:

```bash
!python -m rx_recognition.train_yolo --data /kaggle/working/Medi-Script-AI/ml/doctors-prescriptions-handwriting-1/data.yaml --epochs 50
```

> **Note:** Adjust the path to `data.yaml` if Roboflow saves it in a slightly different directory!

When training finishes, you can download your brand new, highly-accurate `best.pt` weights file! It will be located at:
`runs/detect/yolov8_custom/weights/best.pt`

---

## 4. Training TrOCR (Recognition)

> [!WARNING]
> **Dataset Format Requirement**
> The Roboflow `yolov8` export format only contains bounding boxes for Object Detection. It does **not** contain the actual string text (e.g., "Amoxicillin") required to teach TrOCR how to read.

To train TrOCR, you need an **OCR Dataset** (a CSV file containing paths to cropped images of handwriting, alongside the actual ground-truth text they contain).

If you have exported an OCR dataset from Roboflow or Kaggle, you can format it into a `train_manifest.csv` with two columns: `image_path` and `transcription`.

Once you have that manifest, you can train TrOCR by running:

```bash
!python -m rx_recognition.train_trocr --train_manifest /path/to/your/train_manifest.csv --val_manifest /path/to/your/val_manifest.csv --epochs 10
```

When it finishes, download the `trocr-rx-finetuned` folder back to your local machine!

---

## 5. Final Local Integration

Once you have downloaded your `best.pt` (YOLO) and `trocr-rx-finetuned` (TrOCR) from Kaggle:
1. Place `best.pt` anywhere on your computer.
2. Open `ml/src/rx_recognition/pipeline.py` and update the `yolo_weights` path to point to your new `best.pt` file.
3. Open `ml/src/rx_recognition/recognition.py` and update the `model_path` on line 77 to point to your new `trocr-rx-finetuned` folder instead of `"microsoft/trocr-base-handwritten"`.

**Run `uv run --project ml python demo.py` one last time, and watch the pipeline flawlessly read a human handwriting sample!**
