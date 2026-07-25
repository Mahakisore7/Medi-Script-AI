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
        # YOLO inference
        results = self.model(image, verbose=False)[0]
        crops = []

        for box in results.boxes:
            field_name = self.model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0]) * 100.0  # Convert to 0-100 scale

            # Crop the image using the bounding box
            crop = image[y1:y2, x1:x2]
            crops.append((field_name, crop, conf))

        return crops


def segment_fields(
    image: np.ndarray, use_yolo: bool = False, yolo_weights: str = "yolov8n.pt"
) -> list[tuple[str, np.ndarray, float]]:
    """
    Placeholder segmentation module.
    Splits the image horizontally into 5 rough chunks corresponding to our 5 fields.
    If use_yolo is True, uses the trained YOLO model instead.
    """
    if use_yolo:
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
