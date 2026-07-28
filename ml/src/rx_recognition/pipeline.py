import os

import numpy as np

from rx_recognition.postprocess import combined_confidence, validate_field
from rx_recognition.preprocessing import preprocess_image
from rx_recognition.recognition import recognize_text
from rx_recognition.record import build_record
from rx_recognition.segmentation import segment_fields


def run_image_pipeline(image: np.ndarray, mode: str = "yolo") -> dict:
    """
    Runs the full end-to-end pipeline.
    """
    # 1. Preprocess
    clean_image = preprocess_image(image)

    # Check if we have trained YOLO weights to use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_weights = os.path.abspath(
        os.path.join(script_dir, "..", "..", "..", "runs", "detect", "kaggle_best.pt")
    )
    print(f"\n[INFO] Loading YOLO model from: {yolo_weights}")
    use_yolo = os.path.exists(yolo_weights)
    if not use_yolo and mode == "yolo":
        print(
            f"[WARNING] Model file not found at {yolo_weights}! Falling back to dummy segmentation."
        )

    # 2. Segment
    fields = segment_fields(clean_image, use_yolo=use_yolo, yolo_weights=yolo_weights, mode=mode)

    # 3. Recognize & 4. Postprocess
    results = {}
    confidences = {}
    for i, (field_name, crop, seg_conf) in enumerate(fields):
        raw_text, rec_conf = recognize_text(crop)
        clean_text, val_conf = validate_field(field_name, raw_text)

        # Calculate unified confidence
        final_conf = combined_confidence(seg_conf, rec_conf, val_conf)

        # If dataset returns the same class (like "handwritten") multiple times,
        # append an index so it doesn't overwrite
        unique_name = f"{field_name}_{i}" if field_name in results else field_name

        results[unique_name] = clean_text
        confidences[unique_name] = round(final_conf, 2)

    # 5. Build Record
    record = build_record(results)

    # Attach confidences to the final record
    record["_confidences"] = confidences
    # For baseline evaluation, we also want the flat dict of fields
    record["_raw_fields"] = results

    return record
