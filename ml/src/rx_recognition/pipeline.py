import numpy as np

from rx_recognition.postprocess import postprocess_text
from rx_recognition.preprocessing import preprocess_image
from rx_recognition.recognition import recognize_text
from rx_recognition.record import build_record
from rx_recognition.segmentation import segment_fields


def run_image_pipeline(image: np.ndarray) -> dict:
    """
    Runs the full end-to-end walking skeleton pipeline.
    """
    # 1. Preprocess
    clean_image = preprocess_image(image)

    # 2. Segment
    fields = segment_fields(clean_image)

    # 3. Recognize & 4. Postprocess
    results = {}
    for field_name, crop in fields:
        raw_text = recognize_text(crop)
        clean_text = postprocess_text(field_name, raw_text)
        results[field_name] = clean_text

    # 5. Build Record
    record = build_record(results)

    # For baseline evaluation, we also want the flat dict of fields
    record["_raw_fields"] = results

    return record
