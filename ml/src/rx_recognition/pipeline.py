import os

import numpy as np

from rx_recognition.postprocess import combined_confidence, validate_field
from rx_recognition.recognition import recognize_text
from rx_recognition.record import build_record
from rx_recognition.segmentation import segment_fields

# Minimum EasyOCR confidence to keep a detection (Step 4: Confidence Filtering)
MIN_OCR_CONFIDENCE = 40.0


def _run_easyocr_on_crop(reader, crop):
    """Run EasyOCR on a single cropped field image and return combined text."""
    results = reader.readtext(crop, detail=1, paragraph=False)
    texts = []
    total_conf = 0
    count = 0
    for bbox, text, conf in results:
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
      - 'yolo+easyocr': YOLO segmentation → EasyOCR recognition per crop (BEST)
      - 'locateanything': LocateAnything API → TrOCR recognition
    """
    clean_image = image

    # Check if we have trained YOLO weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_weights = os.path.abspath(
        os.path.join(script_dir, "..", "..", "..", "runs", "detect", "kaggle_best.pt")
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
            for field_name, crop, seg_conf, _ in fields:
                ocr_text, ocr_conf = _run_easyocr_on_crop(reader, crop)
                # Use lexicon validation for medicine/symptom fields
                clean_text, val_conf = validate_field(field_name, ocr_text)
                final_conf = combined_confidence(seg_conf, ocr_conf, val_conf)

                results[field_name] = clean_text
                confidences[field_name] = round(final_conf, 2)

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
    for i, (field_name, crop, seg_conf, ocr_text) in enumerate(fields):
        if mode == "easyocr":
            raw_text = ocr_text
            rec_conf = seg_conf
            # Step 4: Confidence filtering — skip low-confidence detections
            if rec_conf < MIN_OCR_CONFIDENCE:
                continue
        else:
            raw_text, rec_conf = recognize_text(crop, use_base_model=False)

        if mode in ("text", "easyocr"):
            clean_text = raw_text
            val_conf = 100.0
        else:
            clean_text, val_conf = validate_field(field_name, raw_text)

        final_conf = combined_confidence(seg_conf, rec_conf, val_conf)
        unique_name = f"{field_name}_{i}" if field_name in results else field_name

        results[unique_name] = clean_text
        confidences[unique_name] = round(final_conf, 2)

    print("\n[DEBUG] Raw Text Lines Detected:")
    import pprint

    pprint.pprint(results)

    # 4. Parse if mode is "text" or "easyocr"
    if mode in ("text", "easyocr"):
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
                sorted_lines.append((idx, v))
        sorted_lines.sort(key=lambda x: x[0])
        lines = [v for _, v in sorted_lines if v.strip()]

        parsed = {"name": "", "age": "", "gender": "", "symptoms": "", "medicine": ""}
        full_text = " \n ".join(lines)

        # Age/Gender heuristic (e.g., 19/M)
        age_gender_match = re.search(r"\(?(\d{1,3})\s*[/\-]\s*([A-Za-z])\)?", full_text)
        if age_gender_match:
            parsed["age"] = age_gender_match.group(1)
            parsed["gender"] = age_gender_match.group(2).upper()

        current_section = None
        symptoms = []
        medicines = []

        for i, line in enumerate(lines):
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
