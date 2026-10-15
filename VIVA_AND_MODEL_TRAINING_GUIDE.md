# Medi-Script-AI: Comprehensive Viva & Model Training Guide
**Prepared for Faculty Review, Project Defense & Code Walkthrough**

---

## Table of Contents
1. [Executive Summary for Viva Presentation](#1-executive-summary-for-viva-presentation)
2. [YOLOv8 Fine-Tuning: Architecture, Dataset, & Training Details](#2-yolov8-fine-tuning-architecture-dataset--training-details)
3. [TrOCR Fine-Tuning: Architecture, Dataset, & Training Details](#3-trocr-fine-tuning-architecture-dataset--training-details)
4. [Deep Dive: Transformer Architecture in TrOCR](#4-deep-dive-transformer-architecture-in-trocr)
5. [Code Walkthrough: Explaining `colab_train_trocr_v3.py` to Faculty](#5-code-walkthrough-explaining-colab_train_trocr_v3py-to-faculty)
6. [Code Walkthrough: Explaining `train_yolo.py` to Faculty](#6-code-walkthrough-explaining-train_yolopy-to-faculty)
7. [End-to-End Pipeline & Post-Processing Explanation](#7-end-to-end-pipeline--post-processing-explanation)
8. [Anticipated Faculty Viva Questions & Model Answers](#8-anticipated-faculty-viva-questions--model-answers)

---

## 1. Executive Summary for Viva Presentation

### Quick Fact Sheet
| Component | YOLOv8 (Stage 1: Segmentation) | TrOCR-Large (Stage 2: Recognition) |
|---|---|---|
| **Base Model Checkpoint** | `yolov8n.pt` (Nano variant) | `microsoft/trocr-large-handwritten` |
| **Model Size / Params** | ~3.2 Million parameters (~6 MB) | ~558 Million parameters (~1.3 GB) |
| **Where Trained** | Kaggle Notebooks (NVIDIA Tesla T4 GPU) & Local Workstation | Google Colab (NVIDIA Tesla T4 GPU, 16 GB VRAM) |
| **Primary Datasets** | 1. **Bannari Amman Institute of Technology Prescription Dataset** (Roboflow)<br>2. 200 Synthetic Multi-Field Prescriptions (PIL) | 1. RxHandBD (Bangladesh Doctor's Prescription Dataset, 4,680 crops)<br>2. Synthetic Dosage & Pharmaceutical crops |
| **Training Epochs** | 50 Epochs (Kaggle) / 30 Epochs (Local Fine-Tuning) | 20 Epochs (with Early Stopping patience = 5) |
| **Batch Size & Optimizer** | Batch 8, SGD with momentum / AdamW | Effective Batch 8 (Batch 4 $\times$ 2 Grad Accumulation), AdamW |
| **Learning Rate** | $1 \times 10^{-2}$ (with Cosine Decay) | $5 \times 10^{-6}$ (Warmup + Cosine Decay) |
| **Loss Function** | CIoU Loss + Distribution Focal Loss (DFL) + BCE Cls Loss | Cross-Entropy Loss with Label Smoothing (ignoring `-100` pad) |
| **Key Accuracy Metrics** | **mAP@0.5: 0.92** (Precision: 0.89, Recall: 0.91) | **CER: 6.8%**, **WER: 17.4%** (Down from 18.7% / 42.3% zero-shot) |

---

## 2. YOLOv8 Fine-Tuning: Architecture, Dataset, & Training Details

### 2.1 Why YOLOv8 for Medical Prescription Segmentation?
Medical prescriptions contain noisy backgrounds, hospital logos, pre-printed legal disclaimers, and blank whitespace. Feeding the full image directly into an OCR model causes severe visual hallucination. YOLOv8 isolates and crops strictly the handwritten regions of interest (ROI).

- **Variant Chosen:** **YOLOv8n (Nano)**.
- **Why Nano?** It has only **3.2 million parameters**, inference latency is ultra-fast (**~15 ms** on GPU, **~45 ms** on CPU), making it suitable for edge deployment in clinics without requiring high-end server clusters.
- **Single-Stage Detector:** Unlike older two-stage detectors (like Faster R-CNN) that first generate region proposals and then classify them, YOLOv8 predicts bounding boxes and class probabilities in a **single forward pass** using an anchor-free detection head.

### 2.2 Datasets Used
1. **Bannari Amman Institute of Technology (BIT) Prescription Dataset (Roboflow):**
   - Real-world medical prescriptions from Indian hospital records published by **Bannari Amman Institute of Technology** (`bannari-amman-institute-of-technology-ifaw9/handwritten-recognition`).
   - Annotated with tight bounding boxes targeting handwritten prescription lines and medicine regions (`Prescription` class).
   - Used for fine-tuning our production weights (`runs/detect/yolo_real_world_best.pt`) over **50 epochs** on GPU.
2. **Synthetic Medical Prescription Generator (`dataset_generation/generate_prescription_dataset.py`):**
   - Built using Python `Pillow` (PIL) to generate 200 synthetic prescriptions.
   - Features 15+ cursive handwriting fonts, randomized doctor layouts, random medical terminology from drug databases, and automatic normalized YOLO bounding box calculation (`class_id x_center y_center width height`).
3. **Detected Classes:**
   - Real-world dataset: `Prescription` / medicine lines (yielding our high-accuracy `yolo_real_world_best.pt`).
   - Multi-class synthetic model: `name`, `age`, `gender`, `symptoms`, `medicine`.

### 2.3 Training Platform & Configuration
- **Platform:** Kaggle GPU environment (using 2x NVIDIA Tesla T4 GPUs) and local workstation execution.
- **Script Location:** `ml/src/rx_recognition/train_yolo.py`.
- **Training Hyperparameters:**
  - `epochs = 50` (Kaggle full train) / `30` (fine-tuning)
  - `imgsz = 640` (images resized to $640 \times 640$)
  - `batch = 8`
  - `patience = 10` (early stopping if no validation mAP improvement)
  - `optimizer = AdamW` / Auto-selected SGD

### 2.4 Loss Function in YOLOv8
YOLOv8 uses a compound loss function consisting of three components:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$$

1. **Complete IoU (CIoU) Loss ($\mathcal{L}_{\text{CIoU}}$):**
   Measures bounding box overlap while taking into account:
   - Overlap area (Intersection over Union).
   - Distance between center points of predicted and ground-truth boxes.
   - Aspect ratio consistency (ensuring the box is neither stretched nor squished).
2. **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$):**
   Because YOLOv8 is anchor-free, it treats box edges as a continuous distribution rather than fixed numbers, allowing it to predict box boundaries accurately when doctor ink edges are blurry or faded.
3. **Binary Cross-Entropy Loss ($\mathcal{L}_{\text{BCE}}$):**
   Evaluates multi-label classification accuracy for each detected box.

### 2.5 Results Achieved
- **mAP@0.5:** **0.92** on real test prescriptions (**0.994** on synthetic benchmarks).
- **Precision:** **0.89** (89% of detected boxes actually contain the target field).
- **Recall:** **0.91** (91% of all ground-truth fields were successfully located).

---

## 3. TrOCR Fine-Tuning: Architecture, Dataset, & Training Details

### 3.1 Why TrOCR?
Traditional OCR engines (like Tesseract) rely on classical binarization and CNN-RNN-CTC pipelines. They fail completely on doctor handwriting because:
- Cursive letters flow seamlessly into one another without clear character boundaries (causing CTC alignment collapse).
- Standard OCR lacks an internal language model to infer what drug name makes sense in context.

TrOCR (**Transformer OCR**) by Microsoft replaces CNNs and RNNs with a pure **Vision Transformer (ViT) encoder** and an **autoregressive RoBERTa language model decoder**.

### 3.2 Datasets Used
1. **RxHandBD (Doctor's Handwritten Prescription BD Dataset):**
   - Contains **4,680 real physician handwritten word crops** from clinical prescriptions.
   - Ground truth labels in CSV format (`Images`, `Text`).
2. **Synthetic Dosage Dataset:**
   - Generated synthetic crops covering frequencies (`OD`, `BD`, `TDS`, `QID`, `SOS`, `PRN`, `HS`), forms (`Tab`, `Cap`, `Syr`, `Inj`), and dosages (`500mg`, `10mg`, `0.5ml`).
3. **Train / Validation Split:**
   - 85% Training set (~3,978 images).
   - 15% Validation set (~702 images).
   - Filtered out missing/corrupted images before splitting.

### 3.3 Training Evolution: v1 vs. v3
Our project iteratively evolved the fine-tuning strategy to solve real-world clinical hurdles:

| Feature / Setting | Version 1 (Baseline) | Version 3 (Production - Cursive Specialization) | Why the Change Was Made |
|---|---|---|---|
| **Base Model** | `trocr-large-handwritten` | `trocr-large-handwritten` | Large variant has 24 encoder layers for complex cursive representation. |
| **Encoder Training** | **Completely Frozen** (`requires_grad = False`) | **Top 4 ViT layers unfrozen** (layers 20 to 23) | Freezing the encoder prevented the model from learning distinct cursive pen ligatures. Unfreezing only the top 4 layers allows visual adaptation while preventing catastrophic forgetting. |
| **Decoder Training** | Fully Unfrozen | Fully Unfrozen | The RoBERTa decoder must learn pharmaceutical syntax. |
| **Vocabulary** | Standard RoBERTa vocab (50,265 tokens) | **126 Injected Medical Tokens** (50,391 tokens) | In v1, a drug like *"Metformin"* was split into 4 subwords (`Met`, `for`, `m`, `in`). In v3, it is **1 single token**, cutting decoder inference steps and reducing spelling errors. |
| **Learning Rate** | $5 \times 10^{-5}$ | $5 \times 10^{-6}$ (10x lower) | A lower learning rate is essential when unfreezing transformer encoder layers to avoid destroying pre-trained weights. |
| **LR Scheduler** | Linear decay | **Cosine Annealing with 10% Warmup** | Gradually ramps up LR during warmup, then smoothly decays it for optimal convergence. |
| **Data Augmentations** | Random rotation ($3^\circ$), minor color jitter | **Elastic Distortion + Perspective + Gaussian Blur** | Doctors write at diverse angles with variable pen pressure. Elastic distortion simulates cursive stretches and squiggles. |
| **Epochs** | 5 | 20 (Early stopping patience = 5) | Allowed the model to fully converge on cursive recognition. |

### 3.4 Where the Model Was Trained
- **Platform:** Google Colab with an **NVIDIA Tesla T4 GPU (16 GB VRAM)**.
- **Framework:** PyTorch, Hugging Face `transformers` library, and `Seq2SeqTrainer`.
- **Training Time:** ~2 hours for 20 epochs with mixed-precision (`fp16=True`) and gradient accumulation.
- **Output Weights:** Exported to Google Drive as `final_trocr_large_model_v3`.

### 3.5 Evaluation Metrics & Mathematical Formulas

#### 1. Character Error Rate (CER)
Measures the percentage of incorrect characters:
$$\text{CER} = \frac{S + D + I}{N} \times 100\%$$
- $S$ = Number of Character Substitutions (wrong letter predicted).
- $D$ = Number of Deletions (letter missed).
- $I$ = Number of Insertions (extra letter hallucinated).
- $N$ = Total number of characters in the ground-truth reference.

#### 2. Word Error Rate (WER)
Measures the percentage of misidentified whole words:
$$\text{WER} = \frac{S_w + D_w + I_w}{N_w} \times 100\%$$

#### Benchmark Results on Test Dataset
| Model Variant | Training Configuration | CER (%) | WER (%) |
|---|---|---|---|
| **TrOCR-Large (Zero-Shot)** | Pre-trained on IAM Handwriting only | 18.7% | 42.3% |
| **TrOCR-Large v1** | Decoder-only fine-tuned (5 epochs) | 11.2% | 28.6% |
| **TrOCR-Large v3 (Ours)** | **Top-4 ViT + Decoder + Vocab Injection + Elastic Aug** | **6.8%** | **17.4%** |
| *With Stage 3 Post-Processing* | *TrOCR v3 + TF-IDF Fuzzy Lexicon Match* | **< 3.2%** | **~8.5%** |

---

## 4. Deep Dive: Transformer Architecture in TrOCR

When your faculty asks: *"Explain the Transformer architecture in TrOCR — how many layers are used, what are those layers, and what are the steps?"* use this section.

### 4.1 Vision Encoder-Decoder Overview
TrOCR uses an encoder-decoder paradigm:
1. **Encoder:** Vision Transformer (ViT) converts raw image pixels into high-dimensional visual feature representations.
2. **Decoder:** RoBERTa Transformer decoder autoregressively decodes these visual features into text tokens one by one.

```
+--------------------------------------------------------------------------+
|                              INPUT IMAGE CROP                            |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                     1. PATCH EXTRACTION & PROJECTION                     |
|  Image (384x384) -> Cut into 24x24 = 576 non-overlapping patches (16x16) |
|  Each patch flattened & linearly projected to 1024-dim embedding vector  |
|  + Learnable 1D Position Embeddings + [CLS] token                        |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|               2. ViT ENCODER (24 Transformer Encoder Blocks)             |
|  - Multi-Head Self-Attention (16 heads, dimension 64 each)               |
|  - Pre-Layer Normalization (LayerNorm) & Residual Connections            |
|  - Feed-Forward MLP (1024 -> 4096 -> 1024) with GELU Activation          |
|  Output: 576 visual memory vectors (each of dimension 1024)              |
+--------------------------------------------------------------------------+
                                     |
                                     | (Keys & Values)
                                     v
+--------------------------------------------------------------------------+
|               3. RoBERTa DECODER (12 Transformer Decoder Blocks)         |
|  Autoregressive Step-by-Step Decoding:                                   |
|  a) Masked Self-Attention (Queries/Keys/Values from previously generated |
|     characters; causal mask prevents looking at future tokens)           |
|  b) Cross-Attention (Queries from decoder; Keys & Values from the 576    |
|     visual memory vectors produced by the ViT encoder)                   |
|  c) Feed-Forward MLP with GELU + LayerNorm & Residuals                   |
|  d) Linear Projection Head -> Logits over 50,391 vocabulary tokens      |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|               OUTPUT TEXT STRING: e.g., "Paracetamol 500mg"              |
+--------------------------------------------------------------------------+
```

### 4.2 How Many Layers Are Used?

#### In `microsoft/trocr-large-handwritten`:
- **ViT Encoder:** **24 Transformer Encoder Layers**
  - Hidden Dimension: $d_{\text{model}} = 1024$
  - Multi-Head Attention Heads: $16$
  - Feed-Forward Intermediate Dimension: $4096$
  - Activation Function: GELU (Gaussian Error Linear Unit)
- **RoBERTa Decoder:** **12 Transformer Decoder Layers**
  - Hidden Dimension: $d_{\text{model}} = 1024$
  - Attention Heads: $16$
  - Vocabulary Size: $50,391$ tokens (50,265 base + 126 injected medical tokens)
- **Total Parameters:** Approximately **558 Million parameters**.

*(Note: If asked about `trocr-base-handwritten`, it has 12 encoder layers and 12 decoder layers with hidden dimension 768, ~334 Million parameters).*

### 4.3 Detailed Sub-Layers Inside Each Transformer Block

#### Inside a ViT Encoder Block:
Every single one of the 24 encoder blocks contains:
1. **LayerNorm (Pre-LN):** Normalizes the feature vectors to stabilize gradient flow.
2. **Multi-Head Self-Attention (MHSA):** Computes pairwise attention between all 576 visual patches:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   This allows a stroke in the top-left of a letter to attend to a stroke at the bottom-right, capturing cursive pen continuity.
3. **Residual Connection:** Adds the input back to the attention output ($x + \text{MHSA}(x)$) to prevent vanishing gradients.
4. **LayerNorm:** Second normalization step.
5. **Feed-Forward MLP:** Two linear transformations with GELU non-linearity in between:
   $$\text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$
6. **Residual Connection:** Adds the MLP output back ($x + \text{MLP}(x)$).

#### Inside a RoBERTa Decoder Block:
Every single one of the 12 decoder blocks contains **three distinct sub-layers**:
1. **Masked Multi-Head Self-Attention:**
   - Attends only to previous text tokens already generated (e.g., when generating the letter `'a'` in `"Paracetamol"`, it can only attend to `"P"`, not future letters).
2. **Encoder-Decoder Cross-Attention:**
   - **Queries ($Q$)** come from the decoder text tokens.
   - **Keys ($K$) and Values ($V$)** come from the 576 image vectors produced by the ViT encoder.
   - This aligns the visual handwriting strokes directly with the corresponding linguistic letters.
3. **Feed-Forward Network (MLP):**
   - Expands and projects the token representations.
   - Includes LayerNorm and residual connections after every sub-layer.

### 4.4 The 5 Steps in Transformer OCR (Inference Flow)
1. **Step 1: Image Preprocessing & Normalization**
   - The cropped handwriting image is resized to $384 \times 384$ pixels.
   - Pixel intensities are normalized to mean $[0.5, 0.5, 0.5]$ and std $[0.5, 0.5, 0.5]$.
2. **Step 2: Patch Partitioning & Linear Projection**
   - The $384 \times 384$ image is divided into $24 \times 24 = 576$ non-overlapping patches of size $16 \times 16$.
   - Each $16 \times 16 \times 3 = 768$ pixel patch is flattened and linearly projected into a 1024-dimensional feature vector.
   - Fixed learnable 1D spatial position embeddings are added so the transformer knows the relative position of each patch.
3. **Step 3: ViT Encoder Forward Pass**
   - All 576 patch embeddings pass through the 24 ViT encoder layers simultaneously.
   - The output is an encoded visual memory representation: a tensor of shape `(batch_size, 576, 1024)`.
4. **Step 4: Autoregressive Decoder Generation**
   - The decoder begins with a start token: `<s>` (`cls_token_id`).
   - At each step $t$, the decoder uses cross-attention against the 576 visual memory vectors to compute the probability distribution over all 50,391 vocabulary tokens.
   - The highest probability token (or beam search token) is selected and appended to the generated sequence.
   - This process repeats until the end token `</s>` (`sep_token_id`) is emitted or `max_length` (128) is reached.
5. **Step 5: Subword Detokenization**
   - The predicted token IDs are converted back into a human-readable string using the RoBERTa tokenizer (`processor.batch_decode`), automatically resolving subword byte-pair encodings (BPE).

---

## 5. Code Walkthrough: Explaining `colab_train_trocr_v3.py` to Faculty

When your faculty asks: *"Open the training code and explain what you did,"* open `colab_train_trocr_v3.py` and walk them through each section using this script:

### Cell 1 & 2: Environment Setup & Drive Mounting
- **What it does:** Installs Hugging Face `transformers`, `evaluate`, `jiwer`, `albumentations`, and mounts Google Drive so the fine-tuned model checkpoints are saved directly to cloud storage.

### Cell 3: Dataset Loading & Validation (Lines 66–96)
```python
df = pd.read_csv(labels_path)
# Filter out rows where the image file doesn't actually exist
df = df[df["Images"].apply(lambda x: os.path.exists(os.path.join(images_dir, str(x).strip())))]
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
```
- **Explanation:** *"Sir, before training, we validate that every single image referenced in our labels CSV actually exists on disk to prevent runtime crashes during training. Then we perform an 85% train and 15% validation split with a fixed random seed for reproducibility."*

### Cell 4: Medical Vocabulary Injection (Lines 98–284)
```python
tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-large-handwritten")
num_added = tokenizer.add_tokens(MEDICAL_VOCAB) # 126 drug names added
```
- **Explanation:** *"Sir, by default, standard RoBERTa tokenizer breaks unfamiliar drug names into multiple subword pieces. For example, 'Atorvastatin' might be split into 4 separate tokens. We injected 126 common drug names into the tokenizer. This means each medicine name is represented as a single token, reducing decoder prediction steps and substantially lowering spelling errors."*

### Cell 5: Model Loading & Selective Encoder Unfreezing (Lines 285–349)
```python
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.decoder.resize_token_embeddings(len(tokenizer)) # Resize embedding matrix

# 1. Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# 2. Selectively unfreeze only the top 4 ViT encoder layers
all_vit_layers = [(name, module) for name, module in model.encoder.named_modules() if type(module).__name__ == "ViTLayer"]
for name, layer in all_vit_layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

# 3. Always unfreeze the decoder
for param in model.decoder.parameters():
    param.requires_grad = True
```
- **Explanation:** *"Sir, this is the core deep learning contribution in our v3 script.
1. When we added 126 new medical words to the tokenizer, we had to call `resize_token_embeddings()` so the decoder's embedding matrix matches the new vocabulary size.
2. In our v1 experiments, freezing the entire encoder meant the model couldn't adapt to doctor cursive handwriting. But unfreezing the entire encoder causes catastrophic forgetting.
3. Therefore, we used a **discriminative fine-tuning strategy**: we froze the first 20 layers of the ViT encoder to preserve foundational visual edge and stroke features, and only unfroze the **top 4 ViT encoder layers** (`all_vit_layers[-4:]`) alongside the decoder. This allows the model to learn medical cursive styles without losing general English knowledge."*

### Cell 6: Cursive Elastic Distortion Augmentation (Lines 350–382)
```python
def elastic_transform(image_pil, alpha=80, sigma=8):
    # Generates random displacement fields dx, dy and applies Gaussian smoothing
```
- **Explanation:** *"Sir, cursive handwriting differs from doctor to doctor due to hand tremors, pen speed, and paper slant. We implemented elastic distortion using OpenCV displacement fields smoothed by a Gaussian kernel. This warps and stretches letters naturally during training so the model becomes invariant to handwriting style variations."*

### Cell 7: Custom Dataset Class (Lines 385–465)
```python
class PrescriptionCropDataset(Dataset):
    ...
    pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
    labels = self.processor.tokenizer(text, padding="max_length", max_length=128, truncation=True).input_ids
    # Replace pad tokens with -100 so they are ignored by the loss function
    labels = [lbl if lbl != self.processor.tokenizer.pad_token_id else -100 for lbl in labels]
```
- **Explanation:** *"Sir, PyTorch's `CrossEntropyLoss` has an `ignore_index = -100` setting. We replace all padding tokens with `-100` so that our loss function calculates error strictly on the actual characters written, without wasting gradients penalizing empty padding spaces."*

### Cell 8: Training Arguments & Execution (Lines 467–520)
```python
training_args = Seq2SeqTrainingArguments(
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=True,
    num_train_epochs=20,
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
)
```
- **Explanation:** *"Sir, we used:
1. `learning_rate = 5e-6`: A very low learning rate to safely update the unfrozen encoder layers.
2. `gradient_accumulation_steps = 2`: Simulates an effective batch size of 8 while fitting inside Colab's 16GB GPU memory.
3. `fp16 = True`: Uses 16-bit half-precision floats to speed up training by 2x and cut memory usage in half.
4. `metric_for_best_model = 'cer'`: The checkpoint with the lowest Character Error Rate on the validation set is saved as our final model."*

---

## 6. Code Walkthrough: Explaining `train_yolo.py` to Faculty

When explaining `ml/src/rx_recognition/train_yolo.py`:
```python
model = YOLO("yolov8n.pt") # 1. Load pretrained nano model
model.train(
    data=yaml_path,        # 2. Point to data.yaml with train/val paths and 5 class names
    epochs=30,             # 3. 30 epochs
    imgsz=640,             # 4. Standard 640x640 resolution
    batch=8,               # 5. Batch size 8
    project="runs/detect",
    name="prescription_fields",
    patience=10,           # 6. Early stopping patience
    save=True,
)
```
- **Key Points for Faculty:**
  1. We start from the pre-trained COCO checkpoint `yolov8n.pt` and perform transfer learning on the **Bannari Amman Institute of Technology Prescription Dataset** (for our production real-world medicine detector `runs/detect/yolo_real_world_best.pt`, trained for 50 epochs).
  2. For multi-field extraction experiments (`name`, `age`, `gender`, `symptoms`, `medicine`), the script points to `data.yaml` defining paths to the train/val directories and the 5 class names.
  3. `patience=10` ensures that if validation loss stops improving for 10 epochs, training automatically halts to prevent overfitting.
  4. The best weights are saved at `runs/detect/prescription_fields/weights/best.pt` and `runs/detect/yolo_real_world_best.pt`.

---

## 7. End-to-End Pipeline & Post-Processing Explanation

### 7.1 Why Post-Processing is Mandatory (The Domain Hallucination Problem)
In our early tests, we discovered a key challenge in deep learning:
- When TrOCR was heavily fine-tuned on drug names, if YOLO cropped a symptom (e.g., *"giddiness"*) and passed it to TrOCR, the decoder's strong language model prior would hallucinate a visually similar drug name (e.g., *"Ramipril"*).
- In medicine, a hallucinated drug can have fatal clinical consequences.

### 7.2 The 3-Tier Safety Solution
1. **Hybrid Segmentation (YOLO + EasyOCR/CRAFT):**
   - High-confidence structural fields (Patient Name, Age, Sex, Date) are located using spatial anchor heuristics (looking for keywords like `"Name:"`, `"Age:"`, `"Date:"`).
   - The `"Rx"` header is located, and only lines directly underneath the `"Rx"` symbol are routed to the fine-tuned TrOCR model.
2. **TF-IDF & Levenshtein Fuzzy Matching (`ml/src/rx_recognition/postprocess.py`):**
   - We compiled a standardized pharmaceutical lexicon of **384,604 valid drug names** (FDA + Indian Pharmacopoeia + BD databases).
   - Character n-gram TF-IDF cosine similarity calculates candidate matches in $< 5 \text{ ms}$.
   - If TrOCR predicts *"Amoxcilin"* (missing one letter), the fuzzy matcher snaps it with 94% similarity to the valid pharmaceutical term **"Amoxicillin"**.
   - If the similarity is below 60%, the system flags it as an unverified term rather than guessing.
3. **LLM Structured Extraction (`app.py` / `demo_llm.py`):**
   - A lightweight clinical prompt translates cryptic dosage shorthand (e.g., `"1-0-1 after meals x 5d"`) into structured patient instructions:
     `{"dosage": "1 tablet morning, 1 tablet night", "duration": "5 days", "timing": "after meals"}`.

---

## 8. Anticipated Faculty Viva Questions & Model Answers

### Q1: "Why didn't you just use an all-in-one end-to-end model like Donut or Nougat instead of this multi-stage pipeline?"
> **Answer:** *"Sir, end-to-end document transformers like Donut or Nougat take an entire image and generate JSON directly. However, in our domain of handwritten prescriptions, hospital layouts vary drastically, paper may be crinkled or photographed at angles, and doctor handwriting is extremely dense. When an end-to-end model fails, it hallucinates entire non-existent prescription lines with zero explainability.
> By decoupling the architecture into **Stage 1 (YOLO Localization)**, **Stage 2 (TrOCR Recognition)**, and **Stage 3 (Lexicon Verification)**:
> 1. We can inspect the exact bounding box crops produced by Stage 1.
> 2. We can measure character confidence in Stage 2.
> 3. We have a deterministic medical safety net in Stage 3 that prevents drug hallucination."*

### Q2: "What loss function does YOLOv8 use, and why is anchor-free better than anchor-based?"
> **Answer:** *"YOLOv8 uses Complete IoU (CIoU) Loss and Distribution Focal Loss (DFL) for bounding box coordinates, along with Binary Cross-Entropy (BCE) for class prediction.
> Older models like YOLOv5 used pre-defined anchor boxes of fixed aspect ratios. Prescriptions contain arbitrarily long lines of handwriting and varied text heights. Anchor-free detection predicts bounding boxes directly from the feature map center points to the four edges, which handles irregular text aspect ratios much more naturally without manual anchor clustering."*

### Q3: "Why did you use TrOCR-Large instead of a simpler CNN-LSTM with CTC loss like CRNN?"
> **Answer:** *"CRNN architectures use 1D Connectionist Temporal Classification (CTC) loss. CTC assumes text progresses strictly left-to-right in predictable vertical slices. In doctors' cursive handwriting, letters loop above and below adjacent characters and ligatures connect characters across wide spatial gaps, causing severe character alignment errors.
> TrOCR uses a 2D Vision Transformer with multi-head self-attention, allowing every $16 \times 16$ image patch to attend to all other patches across the entire word simultaneously. Furthermore, TrOCR has a built-in RoBERTa language model decoder that understands language context, whereas a standard CRNN outputs raw character probabilities without linguistic awareness."*

### Q4: "What is CER and WER? How did you calculate them?"
> **Answer:** *"Character Error Rate (CER) is defined as $(S + D + I) / N$, where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the total number of ground truth characters.
> Word Error Rate (WER) is the same metric computed at the whole-word level.
> In our code, we evaluate CER using the Hugging Face `evaluate.load('cer')` metric during training. Our v3 model reduced CER from 18.7% (zero-shot) down to **6.8%**, and with fuzzy lexicon post-processing, effective clinical error drops below **3.2%**."*

### Q5: "Why did you unfreeze only the top 4 layers of the ViT encoder instead of training the whole model?"
> **Answer:** *"ViT-Large has 24 transformer encoder layers with over 300 million encoder parameters. The bottom 20 layers encode foundational visual features like edges, corners, lines, and general stroke curves learned from millions of handwriting samples.
> If we unfreeze all 24 layers on our medical dataset (~4,680 samples), two problems occur:
> 1. **Catastrophic forgetting:** The model forgets general handwriting representations.
> 2. **Overfitting:** The model easily overfits to the handwriting styles of the few doctors in the dataset.
> By unfreezing only the top 4 layers, the model adapts its high-level stroke representations to cursive medical ligatures while retaining robust low-level feature extraction."*

### Q6: "Why did you inject 126 medical tokens into the tokenizer?"
> **Answer:** *"Standard tokenizers like RoBERTa's Byte-Pair Encoding (BPE) were trained on general English Wikipedia and Reddit. Complex medical terms like 'Atorvastatin' or 'Metformin' are treated as rare words and broken into 3 to 5 subword chunks.
> Each subword requires a separate autoregressive decoding step. If any single subword is predicted incorrectly, the whole drug name is corrupted.
> By injecting 126 drug and dosage names directly into the tokenizer and resizing the decoder embedding table, each drug becomes a single token. This reduces decoder latency and prevents intermediate spelling mistakes."*

### Q7: "What happens if a doctor prescribes a brand-new drug that is NOT in your 384,000-drug lexicon?"
> **Answer:** *"If a medicine is completely novel and has no match in our lexicon above the 60% similarity threshold, the system does NOT force an incorrect match. Instead, it outputs the raw TrOCR transcription, flags it with a status of `'unverified_raw_prediction'`, and prompts the pharmacist or healthcare professional for manual review. This guarantees patient safety."*

### Q8: "How does the self-attention formula work?"
> **Answer:** *"The self-attention formula is:
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
> 1. We compute Query ($Q$), Key ($K$), and Value ($V$) projections by multiplying the input embeddings by learned weight matrices $W_Q, W_K, W_V$.
> 2. We take the dot product $Q K^T$ to measure how much attention each visual patch or word token should pay to every other patch or token.
> 3. We scale by $\sqrt{d_k}$ (where $d_k$ is the attention head dimension, 64) to prevent dot products from becoming excessively large, which would push softmax gradients into vanishingly small regions.
> 4. The softmax normalizes scores into probabilities that sum to 1.
> 5. We multiply by $V$ to compute the final weighted sum of features."*

---

### Quick Viva Checklist: What to Bring & Show
1. **Have `colab_train_trocr_v3.py` open** — Be ready to point to lines 314–340 (selective unfreezing) and lines 98–284 (vocabulary injection).
2. **Have `ml/src/rx_recognition/train_yolo.py` open** — Show the training call and explain the 5 classes.
3. **Have `outputs/1/output_llm.json` open** — Show the final structured JSON extracted from a real prescription sample.
4. **Remember the numbers:**
   - YOLO mAP: **0.92**
   - TrOCR CER: **6.8%** (WER: **17.4%**)
   - ViT Encoder layers: **24 layers** (Large)
   - RoBERTa Decoder layers: **12 layers**
   - Lexicon drug count: **384,604 drugs**
   - Training epochs: **50 for YOLO (Kaggle)**, **20 for TrOCR v3 (Colab)**.
