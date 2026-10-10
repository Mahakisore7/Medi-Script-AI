# 🏥 Medi-Script AI

> **An end-to-end deep learning pipeline for digitizing handwritten medical prescriptions into structured JSON records.**

Medi-Script AI combines object detection, transformer-based OCR, spatial heuristic parsing, fuzzy lexicon matching, and LLM-powered post-processing to extract all clinically relevant fields from a handwritten prescription photograph — and present them in a clean web dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Medicine Detection** | YOLOv8 detects multiple line-level medicine bounding boxes per prescription |
| **Handwriting OCR** | Gemini 2.5 Flash Vision (primary) + fine-tuned TrOCR-Large v3 (fallback) reads handwritten medicine crops |
| **Full-Page Text Detection** | EasyOCR scans the entire document for printed headers, footers, and demographics |
| **Spatial Anchor Parsing** | Finds "Name", "Age", "Sex", "Date" labels and reads the value immediately to their right |
| **Regex + Position Rules** | Classifies text as DoctorName, HospitalClinicInfo, Date, or Other based on content and spatial location |
| **Fuzzy Lexicon Matching** | Corrects OCR spelling errors against a **384,604-entry** Indian medicine database using TF-IDF cosine similarity |
| **Domain Language Model** | Character trigram LM rescores TrOCR beam search outputs for more accurate medicine names |
| **LLM Other-Field Classifier** | Gemini 2.5 Flash intelligently re-assigns unclassified "Other" text into proper fields |
| **LLM Dosage Advice** | Gemini 2.5 Flash translates cryptic abbreviations (BD, AC, OD, HS) into plain-English patient instructions |
| **Web Dashboard** | FastAPI + HTML/CSS/JS frontend with drag-and-drop upload and beautiful glassmorphism results cards |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Input Prescription Image                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   YOLOv8 Detection    │──── Multiple medicine-line boxes
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Crop Enhancement    │──── Upscale, CLAHE, Adaptive Threshold
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Gemini Vision OCR   │──── Primary: reads each crop
              │   (TrOCR v3 fallback) │──── Fallback: if Gemini fails/UNREADABLE
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Fuzzy Lexicon Match  │──── 384K medicines, TF-IDF + RapidFuzz
              └───────────┬───────────┘
                          │
         ┌────────────────▼────────────────┐
         │   EasyOCR Full-Page Detection   │──── All visible text (headers, etc.)
         └────────────────┬────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │  Regex + Spatial Classification │──── DoctorName, Hospital, Date, Other
         │  Spatial Anchor Parsing         │──── Patient Name, Age, Sex
         └────────────────┬────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │  LLM Other-Field Classifier     │──── Re-assigns unclassified text
         │  LLM Dosage Advice Generator    │──── Plain-English instructions
         └────────────────┬────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Structured JSON     │
              └───────────────────────┘
```

---

## 📁 Project Structure

```
Medi-Script-AI/
├── app.py                          # FastAPI web backend
├── demo_llm.py                     # CLI entry point (full pipeline)
├── demo.py                         # CLI entry point (heuristic only)
├── .env                            # GEMINI_API_KEY goes here
├── pyproject.toml                  # Python dependencies (managed by uv)
│
├── ml/src/
│   ├── rx_recognition_heuristic/
│   │   ├── pipeline.py             # Main pipeline orchestrator
│   │   └── regex_classifier.py     # Regex + spatial position rules
│   │
│   ├── rx_recognition/
│   │   ├── recognition.py          # TrOCR v3 model loading & inference
│   │   ├── crop_enhancer.py        # CLAHE, adaptive threshold, dilation
│   │   ├── postprocess.py          # Fuzzy lexicon matching entry point
│   │   ├── medical_lexicon.py      # TF-IDF cosine similarity matcher (384K meds)
│   │   ├── domain_lm.py            # Character trigram LM for beam rescoring
│   │   └── lexicon_trie.py         # Trie-based lexicon lookup
│   │
│   ├── rx_recognition_multiclass/
│   │   └── gemini_ocr.py           # Gemini 2.5 Flash Vision OCR
│   │
│   └── rx_llm_advice/
│       ├── other_classifier.py     # LLM-based Other-field classification
│       └── dosage_advice.py        # LLM-based dosage advice generation
│
├── web/static/
│   ├── index.html                  # Frontend UI
│   ├── style.css                   # Glassmorphism dark theme
│   └── app.js                      # Upload logic & results rendering
│
├── models/                         # Fine-tuned TrOCR weights (not in git)
│   └── final_trocr_large_model_v3/
├── runs/detect/                    # YOLO weights (not in git)
│   └── yolo_real_world_best.pt
├── data/lexicon/                   # Medicine & symptom CSV files
│   ├── medicines.csv               # 384,604 medicine names
│   └── symptoms.csv                # 35 symptom entries
│
├── colab_train_trocr_v3.py         # Colab/Kaggle TrOCR training script
├── colab_train_trocr_incremental.py
└── colab_generate_synthetic_dosages.py
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Gemini API Key** — get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone & Install

```bash
git clone https://github.com/Mahakisore7/Medi-Script-AI.git
cd Medi-Script-AI
uv sync
```

### 2. Set Up API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

### 3. Download Models

| Model | Where to Place | How to Get |
|---|---|---|
| YOLOv8 weights | `runs/detect/yolo_real_world_best.pt` | Trained on Bannari Amman Institute dataset via Ultralytics |
| TrOCR v3 weights | `models/final_trocr_large_model_v3/` | Fine-tuned on RxHandBD via Colab/Kaggle (see training scripts) |

> **Note:** The base TrOCR model (`microsoft/trocr-large-handwritten`) is auto-downloaded from HuggingFace on first run if no local model exists.

---

## 💻 Usage

### Option A: Web Dashboard (Recommended)

Launch the FastAPI server:

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser. Drag and drop a prescription image to process it.

### Option B: Command-Line Interface

Run the full pipeline (with LLM post-processing):

```bash
uv run python demo_llm.py "path/to/prescription.jpg"
```

Run the heuristic-only pipeline (no LLM, no API calls):

```bash
uv run python demo.py "path/to/prescription.jpg" --mode heuristic
```

---

## 📄 Sample Output

```json
{
  "DoctorName": "Dr: Sanjeev Saxena | M.B.B.S | S.M.o.",
  "Date": "17/7/24",
  "HospitalClinicInfo": "No. 18819 | SENIOR MEDICAL OFFICER | MAHARAO BHIM SINGH HOSPITAL, KOTA | 1-J-26, VIGYAN NAGAR, KOTA 324005 (RAJ)",
  "Medical Text": [
    "Ozad : 100 BC 7 AM",
    "Ambulex\nPathil Prm 1OHHS",
    "placida\nEsotat. HC 10D 6 pm",
    "RozuD.\nAmbulox 1 BD",
    "placida 1OD\n110 1OD 6 F"
  ],
  "Other": "",
  "Patient": {
    "Name": "Kamuna Chouhan",
    "Age": "36",
    "Sex": "F"
  },
  "Dosage Advice": "- Ozad: Take 100 before meals (BC) at 7 AM.\n- Ambulex: Take 1 tablet twice a day (BD).\n- placida: Take 1 once a day (OD).\n- Esotat HC: Take 1 once a day (OD) at 6 PM."
}
```

---

## 🧠 Model Training

### TrOCR Fine-Tuning

The TrOCR model is fine-tuned on [RxHandBD](https://data.mendeley.com/datasets/dsb5r6vskg/3) (5,578 handwritten prescription word images). Training must be done on a GPU (T4 or higher).

```bash
# On Google Colab or Kaggle:
python colab_train_trocr_v3.py
```

Key training hyperparameters:
- `per_device_train_batch_size=2`
- `gradient_checkpointing=True`
- `num_train_epochs=8`
- `learning_rate=4e-5`

### YOLO Training

The YOLO detector is trained on the [Bannari Amman Institute prescription dataset](https://universe.roboflow.com/bannari-amman-institute-of-technology-ifaw9/handwritten-recognition) (357 images, single "Prescription" class).

---

## ⚠️ Known Limitations

- **Gemini Free Tier rate limits**: 15-20 RPM. The pipeline adds 4-second delays between API calls to avoid 429 errors. Processing takes ~20-40 seconds per prescription.
- **Handwriting generalization**: TrOCR v3 may struggle on handwriting styles very different from the RxHandBD training data.
- **EasyOCR accuracy**: Faint or highly cursive patient demographic text may be misread.
- **Standard layouts assumed**: Spatial anchor parsing expects labels on the left and values to the right.

---

## 📚 Tech Stack

| Component | Technology |
|---|---|
| Object Detection | Ultralytics YOLOv8 |
| Handwriting OCR (Primary) | Gemini 2.5 Flash Vision API |
| Handwriting OCR (Fallback) | Fine-tuned TrOCR-Large v3 (HuggingFace Transformers) |
| Full-Page Text Detection | EasyOCR (CRNN) |
| Fuzzy Matching | RapidFuzz + scikit-learn TF-IDF |
| LLM Post-Processing | Google Gemini 2.5 Flash API |
| Web Backend | FastAPI + Uvicorn |
| Web Frontend | Vanilla HTML/CSS/JS |
| Package Manager | uv |

---

## 👥 Authors

- **Mahakisore M** — CB.SC.U4AIE24333
- **Yashwanth B** — CB.SC.U4AIE24360

Department of Computer Science and Engineering (Artificial Intelligence)
Deep Learning — Course Code: 22AIE304

---

## 📝 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
