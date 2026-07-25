import numpy as np


def segment_fields(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Placeholder segmentation module.
    Splits the image horizontally into 5 rough chunks corresponding to our 5 fields.
    This will be replaced by Mask R-CNN in Phase 3.
    Returns a list of (field_name, image_crop).
    """
    height, width = image.shape[:2]
    step = height // 5

    fields = ["name", "age", "gender", "symptoms", "medicine"]
    crops = []

    for i, field in enumerate(fields):
        y1 = i * step
        y2 = height if i == 4 else (i + 1) * step
        crop = image[y1:y2, 0:width]
        crops.append((field, crop))

    return crops
