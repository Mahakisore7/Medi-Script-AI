import os
from typing import Any

import numpy as np

from rx_recognition.recognition import recognize_text
from rx_recognition_heuristic.regex_classifier import classify_text_heuristic

# Minimum confidence to accept OCR results
MIN_OCR_CONFIDENCE = 15.0


def _get_yolo_boxes(image: np.ndarray, yolo_weights: str, label_path: str = None) -> list:
    """
    Runs YOLO to find the prescription regions (often multiple line-level boxes).
    If a label_path is provided (for testing), it reads the exact ground truth box instead!
    Returns a list of (x1, y1, x2, y2) tuples.
    """
    if label_path and os.path.exists(label_path):
        print(f"[INFO] Using ground truth label file for perfect crop: {label_path}")
        h_img, w_img = image.shape[:2]
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_c, y_c, w, h = map(float, parts[1:5])
                    x_c, y_c, w, h = x_c * w_img, y_c * h_img, w * w_img, h * h_img
                    x1 = int(x_c - (w / 2))
                    y1 = int(y_c - (h / 2))
                    x2 = int(x_c + (w / 2))
                    y2 = int(y_c + (h / 2))
                    boxes.append((x1, y1, x2, y2))
        return boxes

    from ultralytics import YOLO

    if not os.path.exists(yolo_weights):
        print(f"[WARNING] YOLO weights not found at {yolo_weights}")
        return []

    model = YOLO(yolo_weights)
    results = model(image, verbose=False, conf=0.25)[0]

    boxes = []
    for box in results.boxes:
        boxes.append(tuple(map(int, box.xyxy[0].tolist())))

    return boxes


def is_overlapping_any(box_inner, yolo_boxes, threshold=0.1):
    """
    Checks if box_inner overlaps with ANY of the yolo_boxes.
    box format: [x1, y1, x2, y2]
    """
    for y_box in yolo_boxes:
        ix1 = max(box_inner[0], y_box[0])
        iy1 = max(box_inner[1], y_box[1])
        ix2 = min(box_inner[2], y_box[2])
        iy2 = min(box_inner[3], y_box[3])

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        inner_area = (box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1])
        if inner_area > 0 and (inter_area / inner_area) > threshold:
            return True

    return False


def run_heuristic_pipeline(
    image: np.ndarray, yolo_weights_path: str = None, label_path: str = None, output_dir: str = None
) -> dict[str, Any]:
    """
    1. Finds Prescription Anchor via YOLO (or ground truth label if provided).
    2. Finds all text via EasyOCR.
    3. Reconciles and classifies.
    """
    import easyocr
    import torch

    if yolo_weights_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_weights_path = os.path.abspath(
            os.path.join(script_dir, "..", "..", "..", "runs", "detect", "yolo_real_world_best.pt")
        )

    print("[INFO] Starting Heuristic Pipeline...")

    final_parsed = {
        "DoctorName": [],
        "PatientInfo": [],
        "Date": [],
        "HospitalClinicInfo": [],
        "Medical Text": [],
        "Other": [],
    }

    h_img, w_img = image.shape[:2]

    # 1. Anchor Detection (Now a list of boxes)
    yolo_boxes = _get_yolo_boxes(image, yolo_weights_path, label_path)

    if output_dir:
        import cv2

        yolo_img = image.copy()
        for box in yolo_boxes:
            cv2.rectangle(yolo_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, "yolo_crops.jpg"), yolo_img)

    if yolo_boxes:
        print(f"[INFO] YOLO found {len(yolo_boxes)} Prescription crops!")
    else:
        print("[WARNING] No YOLO crops found! Will classify everything via regex.")

    # 2. Process YOLO crops (Prescription / Medical Text)
    for i, y_box in enumerate(yolo_boxes):
        print(f"[TrOCR] Processing YOLO crop {i+1}/{len(yolo_boxes)} as Prescription...")
        # Add slight padding to the YOLO box for better OCR
        pad = 5
        c_x1 = max(0, y_box[0] - pad)
        c_y1 = max(0, y_box[1] - pad)
        c_x2 = min(w_img, y_box[2] + pad)
        c_y2 = min(h_img, y_box[3] + pad)
        yolo_crop = image[c_y1:c_y2, c_x1:c_x2]

        if yolo_crop.size > 0:
            try:
                from rx_recognition.crop_enhancer import enhance_crop_for_ocr

                yolo_crop = enhance_crop_for_ocr(yolo_crop, scale=2.0)
            except Exception as e:
                print(f"[WARNING] Crop enhancement failed: {e}")

            from rx_recognition.postprocess import correct_medicine_crop

            text, conf = recognize_text(yolo_crop, use_base_model=False, field_type="medicine")

            # Apply Fuzzy Lexicon Matching to correct minor OCR spelling mistakes
            corrected_text, fuzzy_conf = correct_medicine_crop(text)
            if corrected_text != text:
                print(f"  [Fuzzy] Corrected '{text}' -> '{corrected_text}'")

            final_parsed["Medical Text"].append(corrected_text)

    # 3. Global Text Detection (EasyOCR)
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=use_gpu)
    print("[INFO] Running Global Text Detection for headers/footers...")
    ocr_results = reader.readtext(image, detail=1, paragraph=False)

    if output_dir:
        import cv2

        easyocr_img = image.copy()
        for res in ocr_results:
            if len(res) == 2:
                bbox = res[0]
            else:
                bbox = res[0]
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(easyocr_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(output_dir, "easyocr.jpg"), easyocr_img)

    # 4. Reconciliation and Spatial Parsing
    non_med_boxes = []

    for result in ocr_results:
        if len(result) == 2:
            bbox, text = result
        else:
            bbox, text, conf = result
            if conf * 100.0 < MIN_OCR_CONFIDENCE:
                continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        if is_overlapping_any([x1, y1, x2, y2], yolo_boxes):
            continue

        non_med_boxes.append(
            {
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
            }
        )

        y_center_norm = ((y1 + y2) / 2) / h_img
        x_center_norm = ((x1 + x2) / 2) / w_img
        label = classify_text_heuristic(text, y_center_norm, x_center_norm)
        if label not in ["PatientInfo"]:
            final_parsed[label].append(text)
            print(f"  [Regex] '{text}' -> {label}")

    # --- SPATIAL PARSING FOR PATIENT INFO ---
    patient = {"Name": "", "Age": "", "Sex": ""}
    date_val = ""

    def find_closest_right(anchor_box, boxes, max_y_diff=30):
        best_box = None
        min_dist = 9999
        for b in boxes:
            if b == anchor_box:
                continue
            if abs(b["cy"] - anchor_box["cy"]) < max_y_diff and b["x1"] >= anchor_box["x2"]:
                dist = b["x1"] - anchor_box["x2"]
                if dist < min_dist:
                    min_dist = dist
                    best_box = b
        return best_box

    def find_all_right(anchor_box, boxes, max_y_diff=30, max_x=9999):
        found = []
        for b in boxes:
            if b == anchor_box:
                continue
            if (
                abs(b["cy"] - anchor_box["cy"]) < max_y_diff
                and b["x1"] >= anchor_box["x2"]
                and b["x2"] < max_x
            ):
                found.append(b)
        found.sort(key=lambda x: x["x1"])
        return " ".join([f["text"] for f in found])

    from rapidfuzz import fuzz

    def is_fuzzy_match(keywords, text, threshold=75):
        if not text:
            return False
        words = text.lower().replace(",", " ").replace(".", " ").split()
        for kw in keywords:
            for w in words:
                if fuzz.ratio(kw, w) >= threshold:
                    return True
        return False

    for b in non_med_boxes:
        low_t = b["text"].lower()
        import re

        if is_fuzzy_match(["name", "patient"], low_t) or re.search(
            r"\b(name|namc|patient)\b", low_t
        ):
            print(f"[DEBUG Spatial] Found Name anchor: '{b['text']}' at {b['cx']}, {b['cy']}")
            patient["Name"] = re.sub(r"(?i)namc|name|patient|nime", "", b["text"]).strip(" :_")
            if not patient["Name"] or len(patient["Name"]) < 3:
                # Find the 'Sex' anchor to know where to stop
                stop_x = 9999
                for cand in non_med_boxes:
                    if (
                        re.search(r"\b(sex|gender|gen)\b", cand["text"].lower())
                        and abs(cand["cy"] - b["cy"]) < 25
                    ):
                        stop_x = cand["x1"]
                        break

                full_name = find_all_right(b, non_med_boxes, max_y_diff=15, max_x=stop_x)
                if full_name:
                    patient["Name"] = full_name
                    print(f"  -> Extracted full name: '{full_name}'")

        elif is_fuzzy_match(["age", "years"], low_t) or re.search(r"\b(age|ag|years|yrs)\b", low_t):
            print(f"[DEBUG Spatial] Found Age anchor: '{b['text']}' at {b['cx']}, {b['cy']}")
            m = re.search(r"(\d{1,3})", b["text"])
            if m:
                patient["Age"] = m.group(1)
            else:
                right = find_closest_right(b, non_med_boxes)
                if right:
                    print(f"  -> Closest right box: '{right['text']}'")
                    age_text = right["text"]
                    # Fix common typo where 'yr' is read as '7' when merged (e.g. 47 / F -> 4 yr / F)
                    if re.search(r"(\d)7\s*/\s*[FMfm]", age_text):
                        age_text = re.sub(r"(\d)7\s*/\s*([FMfm])", r"\1 yr / \2", age_text)
                        print(f"     -> Corrected age typo: '{age_text}'")
                    m2 = re.search(r"(\d{1,3})", age_text)
                    if m2:
                        patient["Age"] = m2.group(1)

        elif is_fuzzy_match(["sex", "gender"], low_t) or re.search(r"\b(sex|gender|gen)\b", low_t):
            print(f"[DEBUG Spatial] Found Sex anchor: '{b['text']}' at {b['cx']}, {b['cy']}")
            if re.search(r"\b(f|female)\b", low_t):
                patient["Sex"] = "F"
            elif re.search(r"\b(m|male)\b", low_t):
                patient["Sex"] = "M"
            else:
                right = find_closest_right(b, non_med_boxes, max_y_diff=45)
                if right:
                    print(f"  -> Closest right box: '{right['text']}'")
                    if re.search(r"\b(f|female)\b", right["text"].lower()):
                        patient["Sex"] = "F"
                    elif re.search(r"\b(m|male)\b", right["text"].lower()):
                        patient["Sex"] = "M"
                    # Handle faint handwriting: sometimes 'F' is read as something weird but it's a single char
                    elif len(right["text"].strip()) <= 2 and right["text"].lower().startswith("f"):
                        patient["Sex"] = "F"
                    elif len(right["text"].strip()) <= 2 and right["text"].lower().startswith("m"):
                        patient["Sex"] = "M"

            if not patient["Sex"]:
                print("  -> Sex not found via EasyOCR. Falling back to TrOCR crop.")
                stop_x = w_img
                for cand in non_med_boxes:
                    if (
                        re.search(r"\b(age|ag|date)\b", cand["text"].lower())
                        and cand["x1"] > b["x2"]
                        and abs(cand["cy"] - b["cy"]) < 30
                    ):
                        stop_x = cand["x1"]
                        break

                c_x1 = b["x2"]
                c_x2 = min(stop_x, b["x2"] + 150)
                c_y1 = max(0, b["y1"] - 10)
                c_y2 = min(h_img, b["y2"] + 10)
                sex_crop = image[c_y1:c_y2, c_x1:c_x2]
                if sex_crop.size > 0:
                    try:
                        stext, sconf = recognize_text(
                            sex_crop, use_base_model=False, field_type="medicine"
                        )
                        print(f"     -> TrOCR saw: '{stext}'")
                        if (
                            re.search(r"\b(f|female)\b", stext.lower())
                            or "F" in stext
                            or "f" in stext
                        ):
                            patient["Sex"] = "F"
                        elif (
                            re.search(r"\b(m|male)\b", stext.lower())
                            or "M" in stext
                            or "m" in stext
                        ):
                            patient["Sex"] = "M"
                    except Exception as e:
                        print(f"     -> TrOCR fallback failed: {e}")

        # Fallback 1: Implicit Name Anchor (Mr., Mrs., Ms., Mast., Master)
        elif not patient.get("Name") and re.search(
            r"\b(mr\.?|mrs\.?|ms\.?|mast\.?|master)\b", low_t
        ):
            print(
                f"[DEBUG Spatial] Found implicit Name anchor (Title): '{b['text']}' at {b['cx']}, {b['cy']}"
            )
            patient["Name"] = re.sub(
                r"(?i)\b(mr\.?|mrs\.?|ms\.?|mast\.?|master)\b", "", b["text"]
            ).strip(" :_")
            if not patient["Name"] or len(patient["Name"]) < 3:
                full_name = find_all_right(b, non_med_boxes, max_y_diff=15)
                if full_name:
                    patient["Name"] = full_name
                    print(f"  -> Extracted full name via title: '{full_name}'")

        # Fallback 2: Implicit Age/Sex Pattern (e.g. 28/M, 45 / F, 12yrs/m)
        elif not (patient.get("Age") and patient.get("Sex")):
            m_age_sex = re.search(r"(\d{1,3})\s*(?:yrs|yr|y|years)?\s*/\s*([mMfF])\b", b["text"])
            if m_age_sex:
                print(f"[DEBUG Spatial] Found implicit Age/Sex pattern: '{b['text']}'")
                if not patient.get("Age"):
                    patient["Age"] = m_age_sex.group(1)
                if not patient.get("Sex"):
                    patient["Sex"] = m_age_sex.group(2).upper()

        # Fallback 3: Implicit Date Pattern (e.g. 12/10/22, 12-10-2022) - allowing some OCR alphanumeric errors
        if not date_val:
            m_date = re.search(
                r"\b([0-9A-Za-z]{1,2}\s*[./-]\s*[0-9A-Za-z]{1,2}\s*[./-]\s*[0-9A-Za-z]{2,4})\b",
                b["text"],
            )
            if m_date:
                # Basic check to avoid matching random things like "O/whitetuskdental" (already handled by bounded length)
                print(f"[DEBUG Spatial] Found implicit Date pattern: '{m_date.group(1)}'")
                date_val = m_date.group(1)

        if is_fuzzy_match(["date"], low_t) or re.search(r"\b(date|dt)\b", low_t):
            print(f"[DEBUG Spatial] Found Date anchor: '{b['text']}' at {b['cx']}, {b['cy']}")
            date_val = re.sub(r"(?i)date|dt", "", b["text"]).strip(" :_")
            if not date_val or len(date_val) < 4:
                # People write dates hastily, often floating above or dipping below the line.
                # Let's search all boxes to the right within a generous Y-band (70px)
                # and explicitly look for something that resembles a date (contains / or -)
                best_date_box = None
                min_dist = 9999
                for cand in non_med_boxes:
                    if cand == b:
                        continue
                    if abs(cand["cy"] - b["cy"]) < 75 and cand["x1"] >= b["x2"]:
                        dist = cand["x1"] - b["x2"]
                        # If it contains typical date chars, we heavily prefer it
                        is_date_like = bool(re.search(r"[/.\-]", cand["text"])) and any(
                            c.isdigit() for c in cand["text"]
                        )
                        effective_dist = dist - 1000 if is_date_like else dist
                        if effective_dist < min_dist:
                            min_dist = effective_dist
                            best_date_box = cand

                if best_date_box:
                    date_val = best_date_box["text"]
                    print(f"  -> Closest date-like right box: '{best_date_box['text']}'")

    # Fix common EasyOCR Date typos (e.g., misreading '/' as '1' -> '1717/24' instead of '17/7/24')
    if date_val:
        # If we see DD1M/YY or D1MM/YY where a slash is expected:
        import re

        # Replaces a '1' that is sandwiched between valid day/month and a trailing slash/year
        date_val = re.sub(r"(\d{1,2})1(\d{1,2}[/\-]\d{2,4})", r"\1/\2", date_val)

    final_parsed["Patient"] = patient
    if date_val:
        final_parsed["Date"] = date_val
    elif final_parsed.get("Date"):
        final_parsed["Date"] = " | ".join(final_parsed["Date"])
    else:
        final_parsed["Date"] = ""

    if "PatientInfo" in final_parsed:
        del final_parsed["PatientInfo"]

    # Cleanup empty fields and join
    for k in list(final_parsed.keys()):
        if k in ["Patient", "Date", "Medical Text"]:
            continue
        if isinstance(final_parsed[k], list):
            final_parsed[k] = " | ".join(final_parsed[k])

    return final_parsed
