import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from rx_recognition.field_mapping import map_text_regions_to_fields
from rx_recognition.postprocess import combined_confidence, validate_field
from rx_recognition.preprocessing import preprocess_image
from rx_recognition.recognition import recognize_text
from rx_recognition.record import build_record
from rx_recognition.segmentation import segment_fields


def _save_debug_artifacts(
    original_image: np.ndarray,
    clean_image: np.ndarray,
    debug_items: list[dict[str, object]],
    results_summary: list[dict[str, object]],
    final_record: dict,
    debug_dir: str,
) -> None:
    output_dir = Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "original.png"), original_image)
    cv2.imwrite(str(output_dir / "preprocessed.png"), clean_image)

    if len(original_image.shape) == 2:
        boxed_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        boxed_image = original_image.copy()

    for idx, item in enumerate(debug_items):
        x1, y1, x2, y2 = item["bbox"]
        field_name = str(item["field_name"])
        seg_conf = float(item["seg_conf"])
        detector = str(item["detector"])
        label = f"{field_name} | {detector} | {seg_conf:.1f}"

        cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            boxed_image,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        crop_path = output_dir / f"crop_{idx:02d}_{field_name}.png"
        cv2.imwrite(str(crop_path), item["crop"])

    cv2.imwrite(str(output_dir / "detected_boxes.png"), boxed_image)

    with open(output_dir / "field_debug.json", "w", encoding="utf-8") as file:
        json.dump(results_summary, file, indent=2)

    with open(output_dir / "final_record.json", "w", encoding="utf-8") as file:
        json.dump(final_record, file, indent=2)


def run_image_pipeline(
    image: np.ndarray,
    mode: str = "yolo",
    debug_dir: str | None = None,
    dbnet_model_path: str | None = None,
) -> dict:
    """
    Runs the full end-to-end pipeline.
    """
    original_image = image.copy()

    # 1. Preprocess
    clean_image = preprocess_image(image)
    segmentation_input = original_image if mode == "dbnet" else clean_image

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
    fields, debug_items = segment_fields(
        segmentation_input,
        use_yolo=use_yolo,
        yolo_weights=yolo_weights,
        mode=mode,
        return_debug=True,
        dbnet_model_path=dbnet_model_path,
    )

    # 3. Recognize & 4. Postprocess
    results = {}
    confidences = {}
    results_summary = []
    for i, (field_name, crop, seg_conf) in enumerate(fields):
        ocr_input = preprocess_image(crop) if mode == "dbnet" else crop
        raw_text, rec_conf = recognize_text(ocr_input)
        clean_text, val_conf = validate_field(field_name, raw_text)

        # Calculate unified confidence
        final_conf = combined_confidence(seg_conf, rec_conf, val_conf)

        # If dataset returns the same class (like "handwritten") multiple times,
        # append an index so it doesn't overwrite
        unique_name = f"{field_name}_{i}" if field_name in results else field_name

        results[unique_name] = clean_text
        confidences[unique_name] = round(final_conf, 2)
        debug_items[i]["crop"] = crop
        debug_items[i]["raw_text"] = raw_text
        debug_items[i]["clean_text"] = clean_text
        debug_items[i]["rec_conf"] = rec_conf
        debug_items[i]["val_conf"] = val_conf
        debug_items[i]["final_conf"] = round(final_conf, 2)
        debug_items[i]["unique_name"] = unique_name
        results_summary.append(
            {
                "unique_name": unique_name,
                "field_name": field_name,
                "detector": debug_items[i]["detector"],
                "bbox": debug_items[i]["bbox"],
                "seg_conf": round(seg_conf, 2),
                "rec_conf": round(rec_conf, 2),
                "val_conf": round(val_conf, 2),
                "final_conf": round(final_conf, 2),
                "raw_text": raw_text,
                "clean_text": clean_text,
            }
        )

    # 5. Build Record
    if mode == "dbnet":
        mapped_fields = map_text_regions_to_fields(results_summary)
        record = build_record(mapped_fields)
        record["_mapped_fields"] = mapped_fields
    else:
        record = build_record(results)

    # Attach confidences to the final record
    record["_confidences"] = confidences
    # For baseline evaluation, we also want the flat dict of fields
    record["_raw_fields"] = results
    record["_debug_items"] = results_summary

    if debug_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = str(Path("runs") / "debug" / f"pipeline_{timestamp}")

    _save_debug_artifacts(
        original_image=original_image,
        clean_image=clean_image,
        debug_items=debug_items,
        results_summary=results_summary,
        final_record=record,
        debug_dir=debug_dir,
    )
    record["_debug_dir"] = debug_dir

    return record
