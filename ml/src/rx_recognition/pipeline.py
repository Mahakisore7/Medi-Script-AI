import os

import numpy as np

from rx_recognition.postprocess import combined_confidence, validate_field
from rx_recognition.preprocessing import preprocess_image
from rx_recognition.recognition import recognize_text
from rx_recognition.record import build_record
from rx_recognition.segmentation import segment_fields


def run_image_pipeline(image: np.ndarray) -> dict:
    """
    Runs the full end-to-end pipeline.
    """
    # 1. Preprocess
    clean_image = preprocess_image(image)

    # Check if we have trained YOLO weights to use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_weights = os.path.abspath(
        os.path.join(
            script_dir, "..", "..", "runs", "detect", "yolov8n-smoke-test", "weights", "best.pt"
        )
    )
    use_yolo = os.path.exists(yolo_weights)

    # 2. Segment
    fields = segment_fields(clean_image, use_yolo=use_yolo, yolo_weights=yolo_weights)

    # 3. Recognize & 4. Postprocess
    results = {}
    confidences = {}
    for field_name, crop, seg_conf in fields:
        raw_text, rec_conf = recognize_text(crop)
        clean_text, val_conf = validate_field(field_name, raw_text)
        
        # Calculate unified confidence
        final_conf = combined_confidence(seg_conf, rec_conf, val_conf)
        
        results[field_name] = clean_text
        confidences[field_name] = round(final_conf, 2)

    # 5. Build Record
    record = build_record(results)

    # Attach confidences to the final record
    record["_confidences"] = confidences
    # For baseline evaluation, we also want the flat dict of fields
    record["_raw_fields"] = results

    return record
