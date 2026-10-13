# Novelty analysis: Medi-Script-AI vs. the base paper

**Base paper:** B. Biswas, B. K. Sarkar, A. Mustafi, A. Mitra, *A novel approach of doctors' handwritten medical prescription recognition using a deep neural network and generating digital records of the patients*, Journal of Dynamics and Games, 2026, doi:10.3934/jdg.2026016.

**Short answer.** Yes, there is novelty — but it is **architectural and system-level, not an accuracy win**. We have not measured field-level accuracy on real prescriptions, so we cannot claim to beat the base paper's numbers, and the two are not comparable anyway (different data).

## 1. What the base paper does
1. Capture (scanner, phone, webcam) → grayscale, noise removal, Otsu binarisation, skew correction, morphology, contrast enhancement.
2. Contour-based segmentation into patches; a **CNN–SVM** classifies each patch as printed or handwritten (confusion matrix: 485/500 and 488/500).
3. Printed text → **Tesseract**. Handwritten text → per-attribute **CNN features + MLP** classifiers (name; age with MNIST digits + a finite-state digit segmenter; gender and symptoms with EMNIST-style 47-class MLP; medicine names).
4. Attributes aggregated into **CSV** for EHR.
5. Data: >400,000 proprietary prescriptions; reported accuracies ≈ 91 % name, ≈ 89–95 % age, ≈ 89–94 % gender, ≈ 91–97 % symptoms, ≈ 93–95 % medicine, 90.20 % overall.

Observations about the paper itself (for transparency, not criticism of intent): the introduction and conclusion mention an LSTM–CTC module but the described models are CNN + MLP; and the reported numbers differ between the abstract/graphical abstract, the results section and the conclusion (e.g. gender 93.5 % vs 88.6 %; symptoms 90.8 % vs 96.81 %).

## 2. What Medi-Script-AI changes (verified in code)

| # | Improvement | Where in code | Why it matters |
|---|---|---|---|
| 1 | Learned detector (YOLOv8-nano) instead of contours + CNN–SVM for locating handwriting | `pipeline.py::_get_yolo_boxes`, `models/yolo/` | Robust to layout; trains in ~4 min; 6.2 MB |
| 2 | **Line-level sequence recognition** (TrOCR-Large / vision LLM) instead of per-attribute classifiers | `recognition/trocr.py`, `recognition/gemini.py` | Open vocabulary: can produce drug names never seen in training; handles strength + frequency in one pass |
| 3 | **Domain language-model rescoring** of the N-best list | `recognition/domain_lm.py` | Counteracts general-English prior |
| 4 | **Large-lexicon verification** (384,604 names) via TF-IDF char n-grams, dosage-aware tokenisation | `postprocessing/medical_lexicon.py` | Knowledge-based correction; absent from the base design |
| 5 | **Confidence fusion + abstention** (field-level configuration) | `postprocessing/confidence.py` | Empty field instead of fluent wrong drug (Case A) |
| 6 | **Dosage notation** support: synthetic `1-0-1 × 5 days` data, incremental v4 fine-tune, symbol repairs | `training/colab/train_trocr_v4_incremental.py`, `data_gen/synthetic_dosage_colab.py` | Base paper does not model dosage |
| 7 | **Spatial anchor parser** + regex/position classifier for printed/demographic text, with fall-backs | `pipeline.py`, `parsing/regex_classifier.py` | Works on free layouts rather than pre-cropped fields |
| 8 | **Constrained LLM stage** (fills only empty fields; abbreviation expansion, no diagnosis) | `llm/` | Cleans residual text; patient-readable instructions |
| 9 | Tiered fallback → runs offline | `config.py`, `recognition/` | Degrades gracefully without API key |
| 10 | Reproducibility: public data, package, API, Docker, tests | repo | Base paper's data are proprietary |

## 3. What we do **not** improve (and must not claim)
* **Accuracy.** No labelled end-to-end benchmark; only detector metrics (86.1 % mAP50, 46.2 % mAP50–95). Recognition evidence is qualitative.
* **Scale.** 357-image detection set vs >400,000 prescriptions.
* **Printed/handwritten classifier.** The base paper trains and reports one; we sidestep it by detecting handwriting directly and by treating everything outside the boxes as printed.
* **Multilingual.** Base paper claims EMNIST-style extensibility; ours is English only.
* **Independence from external services.** Our best tier calls an external API.
* **Gate coverage.** The abstention gate is not yet in the default page-level path.

## 4. Defensible novelty statement (for the paper)
> We propose a detect–read–verify architecture for handwritten prescriptions that replaces closed-vocabulary glyph/word classification with open-vocabulary line-level recognition, constrains the output with a 384,604-entry lexicon and a domain language model, parses layout-dependent fields with a spatial anchor parser, and exposes a reproducible, offline-capable service. Its design removes four limitations of the CNN–MLP baseline (closed vocabulary, no knowledge verification, no dosage/layout modelling, proprietary data) while its end-to-end accuracy remains to be benchmarked.

## 5. What would turn the claims into evidence (suggested experiments)
1. Transcribe 200–300 real crops; report CER/WER/exact-match for: raw TrOCR v1…v4, + LM rescoring, + lexicon correction, + enhancement, Gemini tier.
2. Evaluate the spatial parser on 50–100 labelled prescriptions (field-level exact match for name/age/sex/date).
3. Reproduce a CNN–MLP baseline on the same crops to enable a like-for-like comparison.
4. Measure CPU latency and API latency per stage.
5. Port the abstention gate to the default path and report coverage vs. error (risk–coverage curve).
