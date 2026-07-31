import logging

import numpy as np


class TrainedFieldDetector:
    """
    Swaps in for segment_fields.
    Uses YOLOv8 to detect field bounding boxes.
    """

    def __init__(self, weights_path: str):
        from ultralytics import YOLO

        # Download nano pretrained if path doesn't exist locally, or use trained weights
        self.model = YOLO(weights_path)

    def detect_fields(self, image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        import cv2
        import os
        
        # YOLO inference with lower confidence threshold
        results = self.model(image, verbose=False, conf=0.25)[0]
        crops = []
        
        # Create debug image copy
        debug_img = image.copy()

        for box in results.boxes:
            field_name = self.model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0]) * 100.0  # Convert to 0-100 scale

            # Crop the image using the bounding box
            crop = image[y1:y2, x1:x2]
            crops.append((field_name, crop, conf))
            
            # Draw bounding box and label for debugging
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{field_name} {conf:.1f}%"
            cv2.putText(debug_img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save the debug image
        script_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "runs", "debug"))
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, "segmentation_debug.jpg")
        cv2.imwrite(debug_path, debug_img)
        logging.info(f"Saved YOLO segmentation debug image to: {debug_path}")

        return crops


class LocateAnythingFieldDetector:
    """
    Experimental segmentation module that interfaces with a Local LocateAnything API Server.
    Instead of relying on trained YOLO weights, it sends the image to the local server
    with text prompts to locate the fields zero-shot.
    """

    def __init__(self, api_url: str = "http://localhost:8000/analyze"):
        self.api_url = api_url
        self.fields_to_prompt = ["name", "age", "gender", "symptoms", "medicine"]

        # NOTE: We only import requests locally here so that if the user doesn't use this mode,
        # they don't need the dependency installed.
        import requests

        self.requests = requests

    def detect_fields(self, image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        import re

        import cv2

        crops = []

        # 1. Encode image to bytes to send via API
        success, encoded_image = cv2.imencode(".jpg", image)
        if not success:
            logging.error("Failed to encode image for LocateAnything API.")
            return crops

        image_bytes = encoded_image.tobytes()
        h, w = image.shape[:2]

        # 2. Single API call - ask model to detect all text regions in the image
        #    LocateAnything returns <ref>text</ref><box><x1><y1><x2><y2></box> pairs
        prompt_text = (
            "Locate all text regions in this medical prescription image. "
            "Find patient name, age, gender, symptoms, and medicine fields."
        )

        try:
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"prompt": prompt_text}

            logging.info("Sending image to LocateAnything API (takes ~2 min on CPU)...")
            response = self.requests.post(self.api_url, files=files, data=data, timeout=300.0)

            if response.status_code == 200:
                result_json = response.json()
                text_result = result_json.get("result", "")
                logging.info(f"LocateAnything raw output: {repr(text_result)}")

                # Parse the actual output format: <ref>label</ref><box><x1><y1><x2><y2></box>
                # Coordinates appear to be in the model's token scale (integers like 118, 272, etc.)
                pairs = re.findall(
                    r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text_result
                )

                if pairs:
                    # Collect all detected boxes with their positions
                    all_boxes = []
                    for label, x1s, y1s, x2s, y2s in pairs:
                        # The model outputs coordinates in its own token space.
                        # We normalize by the model's max coordinate value (1000).
                        n1, n2, n3, n4 = int(x1s), int(y1s), int(x2s), int(y2s)
                        px1 = int((n1 / 1000.0) * w)
                        py1 = int((n2 / 1000.0) * h)
                        px2 = int((n3 / 1000.0) * w)
                        py2 = int((n4 / 1000.0) * h)
                        real_x1, real_x2 = min(px1, px2), max(px1, px2)
                        real_y1, real_y2 = min(py1, py2), max(py1, py2)
                        if real_x2 > real_x1 and real_y2 > real_y1:
                            all_boxes.append(
                                {
                                    "label": label,
                                    "x1": real_x1,
                                    "y1": real_y1,
                                    "x2": real_x2,
                                    "y2": real_y2,
                                }
                            )

                    if all_boxes:
                        # Sort boxes top-to-bottom, then group into 5 fields by Y position
                        all_boxes.sort(key=lambda b: b["y1"])
                        field_names = ["name", "age", "gender", "symptoms", "medicine"]

                        # Merge all boxes into one large bounding region per field bucket
                        # Divide the image height into 5 equal strips and assign boxes to strips
                        strip_h = h // len(field_names)
                        field_groups = {fn: [] for fn in field_names}

                        for box in all_boxes:
                            center_y = (box["y1"] + box["y2"]) // 2
                            field_idx = min(center_y // strip_h, len(field_names) - 1)
                            field_groups[field_names[field_idx]].append(box)

                        for field_name in field_names:
                            boxes_in_field = field_groups[field_name]
                            if not boxes_in_field:
                                continue
                            # Merge all boxes in this field into one crop
                            merged_x1 = min(b["x1"] for b in boxes_in_field)
                            merged_y1 = min(b["y1"] for b in boxes_in_field)
                            merged_x2 = max(b["x2"] for b in boxes_in_field)
                            merged_y2 = max(b["y2"] for b in boxes_in_field)
                            # Add small padding
                            pad = 5
                            merged_x1 = max(0, merged_x1 - pad)
                            merged_y1 = max(0, merged_y1 - pad)
                            merged_x2 = min(w, merged_x2 + pad)
                            merged_y2 = min(h, merged_y2 + pad)
                            crop = image[merged_y1:merged_y2, merged_x1:merged_x2]
                            if crop.size > 0:
                                crops.append((field_name, crop, 75.0))
                                logging.info(
                                    f"  Field '{field_name}': merged {len(boxes_in_field)} boxes → "
                                    f"crop [{merged_x1},{merged_y1},{merged_x2},{merged_y2}]"
                                )
                    else:
                        logging.warning("LocateAnything: all detected boxes had zero area.")
                else:
                    logging.warning(
                        "LocateAnything returned no parseable <ref>/<box> pairs. "
                        f"Raw: {repr(text_result)}"
                    )
            else:
                logging.warning(f"LocateAnything API error: {response.status_code}")

        except Exception as e:
            logging.warning(
                f"Failed to connect to LocateAnything server at {self.api_url}. "
                f"Is it running? Error: {e}"
            )

        return crops


def segment_fields(
    image: np.ndarray, use_yolo: bool = False, yolo_weights: str = "yolov8n.pt", mode: str = "yolo"
) -> list[tuple[str, np.ndarray, float]]:
    """
    Placeholder segmentation module.
    Splits the image horizontally into 5 rough chunks corresponding to our 5 fields.
    If mode is 'yolo', uses the trained YOLO model instead.
    If mode is 'locateanything', uses the local LocateAnything API server.
    """
    if mode == "locateanything":
        detector = LocateAnythingFieldDetector()
        crops = detector.detect_fields(image)
        if len(crops) > 0:
            return crops
        else:
            logging.warning(
                "LocateAnything failed or server offline! Falling back to dummy segmentation."
            )

    elif mode == "yolo" or use_yolo:
        detector = TrainedFieldDetector(yolo_weights)
        crops = detector.detect_fields(image)
        if len(crops) > 0:
            return crops
        else:
            logging.warning("YOLO detected no fields! Falling back to dummy segmentation.")

    height, width = image.shape[:2]
    step = height // 5

    fields = ["name", "age", "gender", "symptoms", "medicine"]
    crops = []

    for i, field in enumerate(fields):
        y1 = i * step
        y2 = height if i == 4 else (i + 1) * step
        crop = image[y1:y2, 0:width]
        # For dummy segmentation, we assign a hardcoded confidence of 50.0
        crops.append((field, crop, 50.0))

    return crops
