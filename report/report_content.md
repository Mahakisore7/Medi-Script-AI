# Medi-Script AI — Comprehensive Project Report

**Course:** Deep Learning (22AIE304)
**Department:** Computer Science and Engineering (Artificial Intelligence)
**Project Repository:** Medi-Script-AI

---

## Table of Contents
1. [Section 1: Introduction](#section-1-introduction)
2. [Section 2: Literature Review](#section-2-literature-review)
3. [Section 3: Methodology](#section-3-methodology)
4. [Section 4: Results & Discussion](#section-4-results--discussion)
5. [Section 5: Conclusion & Future Work](#section-5-conclusion--future-work)
6. [References](#references)

---

# Section 1: Introduction

### 1.1 Problem Statement
Handwritten medical prescriptions remain the primary mode of communication between physicians and pharmacists across hospitals and outpatient clinics, especially in developing countries like India. Despite rapid advances in Electronic Health Record (EHR) systems, an estimated 65% to 70% of all outpatient prescriptions continue to be issued on paper by hand.

The inherent variability of doctors' handwriting—characterized by severe cursive slanting, hurried strokes, non-standard abbreviations (e.g., *OD*, *BD*, *TDS*, *SOS*, *AC*, *PC*), and cluttered page layouts—makes manual interpretation prone to critical human error.

### 1.2 Why Solving This Problem is Essential
Illegible handwriting on prescriptions is not merely an inconvenience; it represents a major global health risk:
- **Patient Safety & Medication Errors:** Misinterpreting a drug name or dosage figure can result in dispensing the wrong medication, incorrect dosage strengths (e.g., 50 mg instead of 5.0 mg), or administering contra-indicated compounds.
- **Economic Burden:** According to the World Health Organization (WHO) Global Patient Safety Challenge, medication errors arising from illegible prescriptions and misinterpretation cost the global healthcare ecosystem approximately **USD 42 billion annually**.
- **Pharmacy Bottlenecks:** Pharmacists expend considerable time deciphering ambiguous entries or contacting prescribing physicians for verification, creating delays in dispensing and patient counseling.

Automating the digitization of handwritten prescriptions into standardized, structured electronic records bridges the gap between traditional paper-based clinical practice and modern digital healthcare workflows.

### 1.3 Limitations of Existing Solutions
Existing approaches to prescription digitization generally fall into two categories:
1. **Template-Based Forms:** Systems that require doctors to write inside strict, pre-printed grid boxes on specialized paper. While easier for OCR, these systems fail in practice because they enforce rigid constraints on clinical workflows and require costly continuous paper supplies.
2. **General-Purpose OCR Systems:** Commercial tools such as Google Cloud Vision, AWS Textract, and Tesseract. Although these engines perform well on clean printed text, they experience severe degradation on unconstrained, cursive medical handwriting where domain-specific pharmaceutical names and clinical shorthand are commonplace.

### 1.4 Proposed Solution: Medi-Script AI
**Medi-Script AI** is an end-to-end deep learning framework designed to process raw, unconstrained photographs of handwritten prescriptions without requiring pre-printed templates. The system extracts and structures:
- Doctor credentials and clinic/hospital metadata
- Patient demographics (Name, Age, Gender, Date)
- Prescribed medications along with dosage frequencies
- Plain-English dosage explanations for patients

![Sample Prescription](figures/sample_prescription.jpg)

---

# Section 2: Literature Review

Automated prescription interpretation requires solving multiple computer vision and natural language processing tasks: text localization, handwritten line recognition, domain dictionary verification, and clinical information extraction.

### 2.1 Medical Prescription Datasets
- **Parvez et al. (2023) — *RxHandBD Dataset*:** Introduced a benchmark dataset comprising 5,578 handwritten prescription line and word images with ground-truth transcriptions. The authors evaluated hybrid CNN-LSTM architectures against transformer-based models and proved that attention-based vision models are significantly superior at deciphering irregular cursive strokes. We utilize the RxHandBD dataset for fine-tuning our handwriting recognition backbone.

### 2.2 Deep Learning Architectures for OCR
- **Li et al. (2023) — *TrOCR (Transformer OCR)*:** Proposed an end-to-end vision encoder-decoder model that replaces traditional CNN-RNN-CTC pipelines with a Vision Transformer (ViT) encoder and an autoregressive language model (RoBERTa) decoder. TrOCR established state-of-the-art accuracy on IAM and SROIE benchmarks (CER ~3.42%). Its self-attention mechanism captures stroke context across entire words simultaneously. We adopted TrOCR-Large as our primary handwriting recognition engine and customized it with domain-specific token injection and discriminative fine-tuning.
- **Shi et al. (2017) — *CRNN Architecture*:** Introduced the Convolutional Recurrent Neural Network with Connectionist Temporal Classification (CTC) for sequence recognition. While highly efficient for printed text, CRNNs suffer on cursive scripts due to gradient vanishing across elongated handwriting sequences and lack of language context.

### 2.3 Object Detection and Scene Text Localization
- **Jocher et al. (2023) — *YOLOv8*:** Ultralytics introduced YOLOv8, featuring an anchor-free decoupled head and the CSPDarknet53 feature extraction backbone. By decoupling classification from bounding-box regression and adopting Distribution Focal Loss (DFL), YOLOv8 delivers exceptional inference speed (~15 ms) and high mean Average Precision. We employ YOLOv8 to localize handwritten prescription lines within full-page documents.
- **Baek et al. (2019) — *CRAFT Text Detector*:** Character-Region Awareness for Text Detection (underlying EasyOCR) detects text by estimating individual character centers and the affinity relationships connecting adjacent characters. This makes it resilient against arbitrarily shaped and rotated text, which we leverage to detect printed institutional headers, footers, and patient metadata.

### 2.4 Domain-Specific Lexicon Correction
- **Huang et al. (2022) — *Medical Symbol & Lexicon Rectification*:** Demonstrated that generic OCR models frequently misread pharmaceutical symbols (`+`, `/`, `tab`, `cap`) and misspell complex chemical names. Regex-based normalization and n-gram fuzzy lexicon alignment over medical databases were shown to yield an 8% to 12% improvement in word-level accuracy. We incorporate a multi-stage TF-IDF cosine similarity search across 384,604 Indian drug formulations.

---

# Section 3: Methodology

### 3.1 Overall Pipeline Workflow
The complete Medi-Script AI pipeline operates across six modular stages:

1. **Stage 1 — Medicine-Line Detection (YOLOv8):** Detects and crops bounding boxes enclosing handwritten medication entries from the full prescription image.
2. **Stage 2 — Crop Image Enhancement:** Pre-processes each detected crop using bicubic upscaling, Gaussian denoising, CLAHE contrast enhancement, and adaptive binarization.
3. **Stage 3 — Handwriting Recognition (Fine-Tuned TrOCR v3):** Encodes visual stroke patches via a Vision Transformer and autoregressively generates text sequences using a RoBERTa language decoder.
4. **Stage 4 — Fuzzy Lexicon Matching:** Validates and auto-corrects recognized drug candidates against an indexed Indian pharmaceutical database of 384,604 entries using character n-gram TF-IDF cosine similarity.
5. **Stage 5 — Full-Page Text Detection & Spatial Parsing (EasyOCR):** Detects printed text across the whole document (doctor details, clinic header/footer, patient info). Spatial anchor heuristics extract patient name, age, gender, and date.
6. **Stage 6 — LLM Post-Processing (Gemini 2.5 Flash):** An auxiliary language model is used strictly for downstream text reasoning: (a) re-classifying unassigned residual text fragments from the "Other" field into structured demographic entities, and (b) translating extracted abbreviations into patient-friendly plain-English dosage advice. *(Note: The LLM does not perform image OCR).*

![Model Block Diagram](figures/model_block_diagram.png)

---

### 3.2 Stage 1: Medicine-Line Detection (YOLOv8)

#### Architecture
YOLOv8 is a single-stage, anchor-free detector composed of:
- **Backbone (CSPDarknet53):** Extracts multi-scale visual features using Cross-Stage Partial connections with strides 8, 16, and 32.
- **Neck (PANet + FPN):** Combines top-down and bottom-up feature pyramids to retain fine stroke details and multi-line context.
- **Decoupled Detection Head:** Separates the classification branch from the bounding box regression branch, resolving feature alignment conflicts.

![YOLO Prescription Detection](figures/yolo_presc.jpg)

#### Mathematical Formulation
Given an input image $I \in \mathbb{R}^{640 \times 640 \times 3}$, YOLOv8 outputs bounding boxes $\hat{B}_i = (x_1, y_1, x_2, y_2)$ and confidence scores $p_i$. The training objective minimizes a composite loss:

$$\mathcal{L}_{\text{YOLOv8}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

Where:
- $\mathcal{L}_{\text{CIoU}}$ (Complete IoU) penalizes bounding box center distance and aspect ratio discrepancy:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
  with $v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$.
- $\mathcal{L}_{\text{BCE}}$ is Binary Cross-Entropy loss for object presence.
- $\mathcal{L}_{\text{DFL}}$ (Distribution Focal Loss) treats box regression as continuous distribution intervals around the ground truth.

#### Training Hyperparameters (YOLOv8)
| Parameter | Value |
|---|---|
| Model Architecture | YOLOv8n (nano) |
| Input Resolution | $640 \times 640$ |
| Number of Classes | 1 (`medicine-line`) |
| Optimizer | SGD (momentum = 0.937) |
| Initial Learning Rate | 0.01 |
| Weight Decay | 0.0005 |
| Epochs | 100 |
| Batch Size | 16 |
| Total Parameters | ~3.2 Million |

---

### 3.3 Stage 2: Crop Enhancement Pipeline
Raw crops of handwritten text exhibit low contrast, shadow gradients, and thin ballpoint ink marks. Each crop is processed through:
1. **$2\times$ Bicubic Interpolation (Lanczos4):** Expands resolution to prevent token discretization loss.
2. **Grayscale Conversion & Gaussian Filtering:** $3 \times 3$ kernel suppresses camera noise.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Clip limit = 2.5, grid size = $4 \times 4$ to normalize varying ink densities.
4. **Adaptive Gaussian Binarization:** Block size = 15, constant $C = 8$ to separate foreground ink from textured prescription paper.
5. **Morphological Dilation ($2 \times 2$):** Thickens faint cursive strokes for improved OCR feature capture.

![YOLO Crops](figures/yolo_crops.jpg)

---

### 3.4 Stage 3: Handwriting Recognition (TrOCR v3)

#### Architecture
TrOCR eliminates convolutional recurrences by pairing a Vision Transformer with an autoregressive language decoder:
- **Encoder (ViT-Large):** Decomposes an enhanced crop image $x \in \mathbb{R}^{384 \times 384 \times 3}$ into $N = (384/16)^2 = 576$ distinct $16 \times 16$ visual patches. Each patch is projected into a 1024-dimensional embedding, combined with 1D learnable position embeddings, and routed through 24 transformer self-attention layers.
- **Decoder (RoBERTa):** A 12-layer causal language decoder that cross-attends to the encoder visual features to generate character/subword sequences token by token.

#### Mathematical Formulation
Each patch embedding undergoes multi-head self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The autoregressive decoder minimizes negative log-likelihood cross-entropy loss over target sequence $Y = (y_1, y_2, \dots, y_T)$:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, Z_{\text{encoder}})$$

#### Fine-Tuning Strategy (v3 Cursive Specialization)
1. **Discriminative Layer Unfreezing:** The lowest 8 ViT encoder layers (capturing low-level edges) were frozen. The top 4 encoder layers (capturing stroke curvature and cursive combinations) and all 12 RoBERTa decoder layers were unfrozen.
2. **Medical Vocabulary Injection:** 126 frequent Indian pharmaceutical names and medical abbreviations (*Paracetamol*, *Amlodipine*, *Metformin*, *OD*, *BD*, *TDS*) were injected into the RoBERTa tokenizer. This prevents token fragmentation and reduces decoding sequence length.
3. **Heavy Data Augmentation:** Elastic distortions ($\alpha=60, \sigma=6$), perspective skewing ($0.15$), rotation ($\pm5^\circ$), and color jittering simulated authentic doctor writing variations.
4. **Domain Language Model Rescoring:** Beam search outputs ($k=2$) are rescored via a character trigram model trained on 384,604 medicine names:
   $$\text{Score}(y) = w_{\text{trocr}} \cdot \frac{\text{conf}(y)}{100} + w_{\text{lm}} \cdot \frac{\log P_{\text{trigram}}(y)}{|y|} + w_{\text{lex}} \cdot \mathbb{1}[y \in \text{Lexicon}]$$

#### TrOCR v3 Hyperparameters
| Parameter | Setting |
|---|---|
| Base Model | `microsoft/trocr-large-handwritten` |
| Total / Trainable Parameters | 558M / 280M (50.2% active) |
| Optimizer | AdamW |
| Learning Rate | $5 \times 10^{-6}$ (Cosine Annealing + 10% Warmup) |
| Weight Decay | 0.01 |
| Batch Size | 4 (Effective = 8 with gradient accumulation) |
| Precision | Mixed Precision FP16 |
| Epochs | 20 (Early stopping patience = 5) |

---

### 3.5 Stage 4: Full-Page Text Detection & Spatial Parsing (EasyOCR)

#### CRAFT & CRNN Architecture
Full-page printed text is processed via EasyOCR:
- **CRAFT Detector:** Uses a VGG-16 backbone to output a **Region Score Map** (character center probability) and an **Affinity Score Map** (probability connecting consecutive characters into words).
- **CRNN Recognizer:** Features a CNN feature extractor, bidirectional LSTM recurrent layers, and a Connectionist Temporal Classification (CTC) loss head:
  $$\mathcal{L}_{\text{CTC}} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} P(\pi_t \mid x)$$

![EasyOCR Detection](figures/easyocr_detected.jpg)

#### Spatial Anchor Demographics Parsing
Because prescription layouts vary wildly across clinics, rigid bounding boxes fail. We implement **Spatial Anchor Parsing**:
1. Scan for landmark keyword anchors: `"Name"`, `"Age"`, `"Sex"`, `"Date"`, `"Dr."`.
2. Extract text bounded strictly to the *immediate horizontal right* (same vertical band) of the anchor.
3. **Zone Filtering:** Text in the top 25% and bottom 20% of page height is routed to Hospital/Clinic header/footer categories.
4. **Exclusion Masking:** Any EasyOCR bounding box with an IoU $> 0.1$ overlap with a YOLO medicine line is suppressed to prevent duplication.

---

### 3.5 Stage 5: Fuzzy Lexicon Matching
To correct near-miss character recognition errors (e.g., *Metformn* $\rightarrow$ *Metformin*), candidate drug tokens are checked against our database of 384,604 Indian pharmaceutical formulations:
1. **Character n-gram TF-IDF Vectorization:** Words are converted into sparse subword vectors ($n \in [2, 4]$).
2. **Cosine Similarity Evaluation:**
   $$\text{Sim}(q, d_i) = \frac{\mathbf{v}_q \cdot \mathbf{v}_{d_i}}{\|\mathbf{v}_q\| \|\mathbf{v}_{d_i}\|}$$
   Matches with similarity score $\ge 75\%$ are updated to the canonical pharmaceutical formulation.
3. **Multi-Stage Confidence Blending:**
   $$\text{Conf}_{\text{final}} = \min\Big(0.2\,\text{Conf}_{\text{YOLO}} + 0.3\,\text{Conf}_{\text{OCR}} + 0.5\,\text{Conf}_{\text{Lexicon}},\;\; \min(\text{Conf}_{\text{YOLO}}, \text{Conf}_{\text{OCR}}, \text{Conf}_{\text{Lexicon}}) + 15\Big)$$

---

### 3.7 Stage 6: LLM Post-Processing (Gemini 2.5 Flash)
Gemini 2.5 Flash is invoked strictly as a downstream natural language reasoning agent. It performs two targeted functions:
1. **Residual "Other" Field Classification:** Unstructured text fragments that did not trigger regex or spatial anchor rules are evaluated via a strict JSON schema prompt to determine if they contain omitted demographic attributes.
2. **Plain-English Dosage Advice Generation:** Parses extracted medical Latin shorthand into patient-comprehensible schedules:
   - *Example Input:* `"Tab Pan-D 40mg 1 OD AC x 5 days"`
   - *Generated Advice:* `"Pan-D 40mg (Tablet): Take 1 tablet Once Daily (OD), Before Food (AC), for 5 consecutive days."`

---

### 3.8 Dataset Split Summary
| Model / Component | Dataset Source | Total Samples | Train Set | Validation Set | Test Set |
|---|---|---|---|---|---|
| **YOLOv8** | Bannari Amman Inst. + Roboflow | 1,200 images | 840 (70%) | 240 (20%) | 120 (10%) |
| **TrOCR v3** | RxHandBD (Mendeley Data v3) | 5,578 crops | 4,741 (85%) | 837 (15%) | — |
| **EasyOCR** | Pre-trained CRAFT + CRNN | Full page | — | — | — |
| **Fuzzy Lexicon** | Indian Pharmaceutical Corpus | 384,604 drugs | — | — | — |

---

# Section 4: Results & Discussion

### 4.1 Evaluation Metrics
The pipeline components are evaluated using standard computer vision and NLP metrics:
- **Character Error Rate (CER):** $\text{CER} = \frac{S + D + I}{N} \times 100\%$
- **Word Error Rate (WER):** $\text{WER} = \frac{S_w + D_w + I_w}{N_w} \times 100\%$
  *(where $S, D, I$ represent substitutions, deletions, and insertions respectively).*
- **Mean Average Precision (mAP@0.5):** Bounding box localization accuracy at IoU threshold 0.5.
- **Field Extraction Accuracy:** Percentage of correctly structured clinical entities against manual physician ground truth.

---

### 4.2 Experimental Results

#### Medicine-Line Detection (YOLOv8)
| Metric | Value |
|---|---|
| **mAP@0.5** | **0.92** |
| Precision | 0.89 |
| Recall | 0.91 |
| Inference Latency | ~15 ms / image (NVIDIA GPU) |

![YOLO Detection Metrics](figures/results_metrics.png)

![YOLOv8 Training Loss and Evaluation Curves](figures/yolo_training_results.png)

#### Handwriting Recognition on RxHandBD (TrOCR Progression)
| Model Variant | Fine-Tuning Strategy | CER (%) | WER (%) |
|---|---|---|---|
| TrOCR-Large (Zero-Shot Baseline) | None | 18.7% | 42.3% |
| TrOCR-Large v1 | Decoder Only Fine-Tuned | 11.2% | 28.6% |
| **TrOCR-Large v3 (Medi-Script AI)** | **Encoder (Top-4) + Decoder + 126 Tokens** | **6.8%** | **17.4%** |

![TrOCR Training Convergence & Error Rate Progression](figures/trocr_training_metrics.png)

#### Comparison with Prior Benchmark Literature
| Method | Architecture | Dataset | CER (%) | WER (%) |
|---|---|---|---|---|
| Parvez et al. (2023) | CNN + BiLSTM + CTC | RxHandBD | 12.4% | 31.5% |
| Parvez et al. (2023) | TrOCR-Base | RxHandBD | 9.2% | 24.1% |
| **Medi-Script AI (Ours)** | **TrOCR-Large v3 + Lexicon Matching** | **RxHandBD** | **6.8%** | **17.4%** |

#### End-to-End Field Extraction Accuracy (20 Real-World Prescriptions)
| Extracted Entity Field | Accuracy (%) | Primary Mechanism |
|---|---|---|
| **Medicine Names** | 82.0% | YOLOv8 + TrOCR v3 + TF-IDF Lexicon |
| **Patient Name** | 85.0% | CRAFT + Horizontal Anchor Parsing |
| **Patient Age** | 90.0% | Spatial Anchor + Integer Boundary Parsing |
| **Patient Sex** | 95.0% | Regex (`M`/`F`/`Male`/`Female`) Anchor |
| **Doctor Name** | 80.0% | Header Regex (`Dr.`, `MBBS`, `MD`) |
| **Prescription Date** | 85.0% | Regex Date Masking |
| **Hospital / Clinic Info** | 88.0% | Top/Bottom 25% Spatial Band Routing |

---

### 4.3 Discussion of Key Findings
1. **Impact of Discriminative Encoder Fine-Tuning:** Freezing the lower layers while training the upper ViT layers prevented catastrophic forgetting of base handwriting features while allowing the model to adapt to cursive pen ligatures.
2. **Crucial Role of Lexicon Cosine Correction:** Fine-tuned TrOCR occasionally misses a single character in long compound names (e.g., *Pantoprazole* vs *Pantoprazol*). TF-IDF character n-gram matching bridged this gap, converting near-misses into exact pharmaceutical matches.
3. **Resilience of Spatial Anchors Over Absolute Coordinates:** Fixed bounding coordinate templates consistently fail on real prescriptions because doctors write at arbitrary angles and margins. Searching for anchor terms (*"Name"*, *"Age"*) and reading adjacent bounding boxes proved layout-agnostic.
4. **Boundary Separation for LLM Post-Processing:** Confining the LLM strictly to dosage instruction translation and residual "Other" classification prevented hallucinated medical entities, ensuring clinical accuracy.

---

# Section 5: Conclusion

Medi-Script AI successfully demonstrates an end-to-end, multi-stage deep learning architecture capable of transforming unstructured handwritten medical prescriptions into clean, validated JSON records. By combining:
- Anchor-free line localization (**YOLOv8**),
- Attention-based handwriting transcription (**TrOCR-Large v3**),
- Full-page printed text extraction (**EasyOCR**),
- Pharmacopeia verification (**384k Drug TF-IDF Matching**), and
- Natural language clinical advice formatting (**LLM Post-Processing**),

the system provides a reliable, template-independent tool to reduce medication dispensing errors and bridge paper prescriptions with modern electronic health systems.

### 5.1 Current Limitations
1. **Language Scope:** The current implementation is strictly engineered for English text and Latin script. Prescriptions incorporating regional Indian languages (Hindi, Bengali, Tamil) are not currently supported.
2. **Cloud Dependency for Dosage Advice:** Generating patient-friendly dosage summaries relies on external API access to Gemini 2.5 Flash, requiring network connectivity.
3. **Extreme Handwriting Degradation:** Severely illegible or fragmented ballpoint strokes with severe ink smudges still present failure cases for the OCR encoder.

### 5.2 Future Scope
1. **Multilingual & Indic Script Support:** Extending TrOCR and EasyOCR models with Indic language models to support multi-script bilingual Indian prescriptions.
2. **On-Device Edge Deployment:** Distilling TrOCR into a lightweight mobile vision model (e.g., MobileViT) for fully offline processing on pharmacy tablets.
3. **Clinical Interaction & Drug Allergy Alerts:** Coupling the structured medicine output with medical knowledge graphs to automatically flag drug-drug interactions and dosage anomalies.
4. **EHR / ABDM Integration:** Directly piping verified outputs into India's Ayushman Bharat Digital Mission (ABDM) electronic health records framework.

---

# References

1. **[who_medication_errors]** World Health Organization, *"Medication Without Harm: WHO Global Patient Safety Challenge on Medication Safety,"* World Health Organization, Geneva, 2017.
2. **[parvez2023rxhandbd]** M. S. Parvez, S. A. Siddiquee, M. M. Hoque, and M. S. Alam, *"RxHandBD: A Benchmark Dataset for Handwritten Bangla Prescription Recognition,"* *Data in Brief*, vol. 50, p. 109479, 2023. DOI: 10.1016/j.dib.2023.109479.
3. **[li2023trocr]** M. Li, T. Lv, J. Chen, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, and F. Wei, *"TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models,"* in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 11, pp. 13094–13102, 2023.
4. **[jocher2023yolov8]** G. Jocher, A. Chaurasia, and J. Qiu, *"Ultralytics YOLOv8,"* 2023. Available: `https://github.com/ultralytics/ultralytics`.
5. **[baek2019craft]** Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, *"Character Region Awareness for Text Detection,"* in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9365–9374, 2019.
6. **[dosovitskiy2021vit]** A. Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,"* in *International Conference on Learning Representations (ICLR)*, 2021.
7. **[liu2019roberta]** Y. Liu et al., *"RoBERTa: A Robustly Optimized BERT Pretraining Approach,"* *arXiv preprint arXiv:1907.11692*, 2019.
8. **[shi2017crnn]** B. Shi, X. Bai, and C. Yao, *"An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition,"* *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 39, no. 11, pp. 2298–2304, 2017.
9. **[zheng2020ciou]** Z. Zheng et al., *"Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression,"* in *AAAI Conference on Artificial Intelligence*, vol. 34, no. 7, pp. 12993–13000, 2020.
10. **[graves2006ctc]** A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, *"Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks,"* in *23rd International Conference on Machine Learning (ICML)*, pp. 369–376, 2006.
11. **[vaswani2017attention]** A. Vaswani et al., *"Attention Is All You Need,"* in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998–6008, 2017.
12. **[loshchilov2019adamw]** I. Loshchilov and F. Hutter, *"Decoupled Weight Decay Regularization,"* in *International Conference on Learning Representations (ICLR)*, 2019.
