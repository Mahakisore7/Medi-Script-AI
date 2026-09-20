import os

import cv2
import numpy as np

from rx_recognition.postprocess import combined_confidence, validate_field
from rx_recognition.recognition import recognize_text
from rx_recognition.record import build_record
from rx_recognition.segmentation import segment_fields

# Minimum final confidence to keep a TrOCR detection in YOLO mode
MIN_OCR_CONFIDENCE = 15.0
# Minimum lexicon match score to accept a medicine crop (filters hallucinations)
MIN_LEXICON_SCORE = 50.0


def _run_easyocr_on_crop(reader, crop):
    """Run EasyOCR on a single cropped field image and return combined text."""
    results = reader.readtext(crop, detail=1, paragraph=False)
    texts = []
    total_conf = 0
    count = 0
    for _bbox, text, conf in results:
        conf_pct = conf * 100.0
        if conf_pct >= MIN_OCR_CONFIDENCE and text.strip():
            texts.append(text.strip())
            total_conf += conf_pct
            count += 1
    avg_conf = total_conf / count if count > 0 else 0
    return " ".join(texts), avg_conf


def run_image_pipeline(image: np.ndarray, mode: str = "yolo") -> dict:
    """
    Runs the full end-to-end pipeline.

    Modes:
      - 'yolo': YOLO segmentation → TrOCR recognition
      - 'text': CRAFT line detection → TrOCR recognition → Fuzzy parsing
      - 'easyocr': EasyOCR detection+recognition → Fuzzy parsing
      - 'yolo+easyocr': YOLO segmentation → EasyOCR recognition per crop
      - 'hybrid': EasyOCR detection+recognition → Parsing → TrOCR on Rx lines (BEST)
      - 'locateanything': LocateAnything API → TrOCR recognition
    """
    clean_image = image

    # Check if we have trained YOLO weights in the new workspace
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_weights = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "..",
            "..",
            "Medicine_Only_Workspace",
            "runs",
            "detect",
            "yolo_real_world_best.pt",
        )
    )
    use_yolo = os.path.exists(yolo_weights)

    # ── HYBRID MODE: YOLO crops → EasyOCR reads each crop ──
    if mode == "yolo+easyocr":
        if not use_yolo:
            print(f"[WARNING] YOLO weights not found at {yolo_weights}!")
            print("[WARNING] Falling back to plain 'easyocr' mode.")
            mode = "easyocr"
        else:
            print(f"\n[INFO] HYBRID MODE: YOLO ({yolo_weights}) -> EasyOCR per crop")
            import easyocr
            import torch

            # Step 1: YOLO segments the image into 5 field crops
            fields = segment_fields(
                clean_image, use_yolo=True, yolo_weights=yolo_weights, mode="yolo"
            )

            # Step 2: Run EasyOCR on each crop
            use_gpu = torch.cuda.is_available()
            reader = easyocr.Reader(["en"], gpu=use_gpu)

            results = {}
            confidences = {}
            from rx_recognition.crop_enhancer import enhance_crop_for_ocr

            # Directory to save crop debug images
            script_dir = os.path.dirname(os.path.abspath(__file__))
            crops_dir = os.path.abspath(
                os.path.join(script_dir, "..", "..", "..", "runs", "debug", "crops")
            )
            os.makedirs(crops_dir, exist_ok=True)
            # Clear old crops from previous runs
            for f in os.listdir(crops_dir):
                if f.endswith(".jpg"):
                    os.remove(os.path.join(crops_dir, f))

            crop_index = 0
            for field_name, crop, seg_conf, _ in fields:
                # Fix 1: Enhance the crop before OCR (upscale + contrast + binarize)
                enhanced = enhance_crop_for_ocr(crop, scale=3.0)
                ocr_text, ocr_conf = _run_easyocr_on_crop(reader, enhanced)

                # Save original and enhanced crop images for visual inspection
                safe_text = "".join(c if c.isalnum() or c in " _-" else "_" for c in ocr_text)[
                    :40
                ].strip()
                cv2.imwrite(
                    os.path.join(crops_dir, f"crop_{crop_index:02d}_original_{safe_text}.jpg"), crop
                )
                cv2.imwrite(
                    os.path.join(crops_dir, f"crop_{crop_index:02d}_enhanced_{safe_text}.jpg"),
                    enhanced,
                )
                crop_index += 1

                # Skip crops where EasyOCR found nothing at all
                if not ocr_text.strip():
                    continue

                # Use lexicon validation for medicine/symptom fields
                clean_text, val_conf = validate_field(field_name, ocr_text)
                final_conf = combined_confidence(seg_conf, ocr_conf, val_conf)

                # Build a unique key so multiple crops of the same class don't overwrite each other
                # e.g. Prescription, Prescription_1, Prescription_2
                unique_name = field_name
                counter = 1
                while unique_name in results:
                    unique_name = f"{field_name}_{counter}"
                    counter += 1

                results[unique_name] = clean_text
                confidences[unique_name] = round(final_conf, 2)

            print(f"\n[DEBUG] Crop images saved to: {crops_dir}")

            print("\n[DEBUG] YOLO+EasyOCR Field Results:")
            import pprint

            pprint.pprint(results)

            record = build_record(results)
            record["_confidences"] = confidences
            record["_raw_fields"] = results
            return record

    # ── STANDARD MODES ──
    print(f"\n[INFO] Loading YOLO model from: {yolo_weights}")
    if not use_yolo and mode == "yolo":
        print(
            f"[WARNING] Model file not found at {yolo_weights}! Falling back to dummy segmentation."
        )

    # 2. Segment
    fields = segment_fields(clean_image, use_yolo=use_yolo, yolo_weights=yolo_weights, mode=mode)

    # 3. Recognize
    results = {}
    confidences = {}
    crops_dict = {}  # Keep track of crops for hybrid mode
    for i, (field_name, crop, seg_conf, ocr_text) in enumerate(fields):
        if mode in ("easyocr", "hybrid", "paddleocr", "paddle_hybrid"):
            raw_text = ocr_text
            rec_conf = seg_conf
            # Step 4: Confidence filtering — skip low-confidence detections
            if rec_conf < MIN_OCR_CONFIDENCE:
                continue
        else:
            base_field_type = field_name.split("_")[0] if "_" in field_name else field_name

            # ── TrOCR-Large Base + Crop Enhancer: The optimal local pure DL setup ──
            try:
                from rx_recognition.crop_enhancer import enhance_crop_for_ocr

                # TrOCR works best with base model on enhanced high-contrast crops
                enhanced_crop = enhance_crop_for_ocr(crop, scale=2.0)
            except Exception as e:
                import logging

                logging.warning(f"Crop enhancement failed: {e}")
                enhanced_crop = crop

            raw_text, rec_conf = recognize_text(
                enhanced_crop, use_base_model=False, field_type=base_field_type
            )
            gemini_succeeded = False  # Always use lexicon correction with TrOCR

        if mode in ("text", "easyocr", "hybrid", "paddleocr", "paddle_hybrid"):
            clean_text = raw_text
            val_conf = 100.0
        elif gemini_succeeded:
            # Gemini already reads accurately — skip fuzzy lexicon correction
            # which was changing correct readings like 'Bisoprolol 2.5mg' → 'Bilacad 2.5mg oral solution'
            clean_text = raw_text
            val_conf = rec_conf  # Trust Gemini's confidence directly
        else:
            # TrOCR fallback — still apply lexicon correction to fix OCR errors
            clean_text, val_conf = validate_field(field_name, raw_text)

        final_conf = combined_confidence(seg_conf, rec_conf, val_conf)

        # ── GATE 1: Skip crops with very low final confidence (hallucinations/noise) ──
        if final_conf < MIN_OCR_CONFIDENCE:
            import logging

            logging.info(
                f"[FILTER] Dropping '{clean_text}' — final_conf {final_conf:.1f} < {MIN_OCR_CONFIDENCE}"
            )
            continue

        # ── GATE 2: For prescription crops, reject text with no plausible medicine match ──
        if field_name.lower().startswith("prescription") and val_conf < MIN_LEXICON_SCORE:
            import logging

            logging.info(
                f"[FILTER] Dropping '{clean_text}' — lexicon score {val_conf:.1f} < {MIN_LEXICON_SCORE} (likely hallucination)"
            )
            continue

        unique_name = f"{field_name}_{i}" if field_name in results else field_name

        results[unique_name] = clean_text
        confidences[unique_name] = round(final_conf, 2)
        crops_dict[unique_name] = crop

    print("\n[DEBUG] Raw Text Lines Detected:")
    import pprint

    pprint.pprint(results)

    # 4. Parse if mode uses line-by-line reading
    if mode in ("text", "easyocr", "hybrid", "paddleocr", "paddle_hybrid"):
        import re

        from rx_recognition.medical_lexicon import (
            correct_medicine_name,
            correct_symptom,
            fuzzy_match_section,
            parse_dosage,
        )

        # Sort lines by index
        sorted_lines = []
        for k, v in results.items():
            if k.startswith("line_"):
                idx = int(k.split("_")[1])
                sorted_lines.append((idx, v, k))
        sorted_lines.sort(key=lambda x: x[0])
        lines = [(v, k) for _, v, k in sorted_lines if v.strip()]

        parsed = {"name": "", "age": "", "gender": "", "symptoms": "", "medicine": ""}
        full_text = " \n ".join([v for v, _ in lines])

        # Age/Gender heuristic (e.g., 19/M)
        age_gender_match = re.search(r"\(?(\d{1,3})\s*[/\-]\s*([A-Za-z])\)?", full_text)
        if age_gender_match:
            parsed["age"] = age_gender_match.group(1)
            parsed["gender"] = age_gender_match.group(2).upper()

        current_section = None
        symptoms = []
        medicines = []

        for _i, (line, line_key) in enumerate(lines):
            first_word = line.split()[0] if line.split() else line
            section_match = fuzzy_match_section(first_word)
            if not section_match and ":" in line:
                section_match = fuzzy_match_section(line.split(":")[0])

            if section_match:
                current_section = section_match
                content = line
                if ":" in line:
                    content = line.split(":", 1)[-1].strip()
                else:
                    content = line.replace(first_word, "").strip()
                if not content:
                    continue
                line = content

            lower_line = line.lower()
            if not current_section:
                if "name" in lower_line:
                    current_section = "Name"
                elif re.search(r"\bc/?o\b", lower_line):
                    current_section = "Symptoms"
                    line = line.split("o", 1)[-1].strip(" :,-")
                elif (
                    re.search(r"\b(mg|tab|cap|syr|iv|stat|ors|fluid|%)\b", lower_line)
                    or "1+1" in lower_line
                    or "1-0" in lower_line
                ):
                    current_section = "Rx"

            if current_section == "Name":
                if not parsed["name"]:
                    name_str = line
                    name_str = re.sub(r"\(?\d{1,3}\s*[/\-]\s*[A-Za-z]\)?", "", name_str).strip()
                    name_str = name_str.replace(".", "").strip()
                    if len(name_str) > 2 and not any(c.isdigit() for c in name_str):
                        parsed["name"] = name_str
            elif current_section == "Symptoms":
                if line:
                    corrected = correct_symptom(line)
                    symptoms.append(corrected)
            elif current_section in ("Rx", "Adv"):
                if line:
                    if mode in ("hybrid", "paddle_hybrid"):
                        # RUN TROCR ON THIS SPECIFIC CROP!
                        crop_img = crops_dict[line_key]
                        print("[DEBUG] Hybrid Mode: Found Rx line. Running TrOCR on crop...")
                        trocr_text, _ = recognize_text(
                            crop_img, use_base_model=False, field_type="medicine"
                        )
                        engine_name = "EasyOCR" if mode == "hybrid" else "PaddleOCR"
                        print(f"        {engine_name} read: '{line}' -> TrOCR read: '{trocr_text}'")
                        line = trocr_text

                    corrected = correct_medicine_name(line)
                    corrected = parse_dosage(corrected)
                    medicines.append(corrected)

        parsed["symptoms"] = ", ".join(symptoms)
        parsed["medicine"] = "\n".join(medicines)

        # Run standard validation/lexicon on the parsed fields
        final_results = {}
        for f, v in parsed.items():
            clean_text, _ = validate_field(f, v)
            final_results[f] = clean_text

        record = build_record(final_results)
        record["_confidences"] = confidences
        record["_raw_fields"] = final_results
    else:
        # 5. Build Record normally
        record = build_record(results)
        record["_confidences"] = confidences
        record["_raw_fields"] = results

    return record
