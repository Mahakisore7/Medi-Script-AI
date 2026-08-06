import os
import logging
import sys
import cv2
from pprint import pprint
from ml.src.rx_recognition.pipeline import run_image_pipeline

# Show INFO logs so we can see LocateAnything output clearly
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    # Allow passing an image path via command line, otherwise default to the sample
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "my sample", "original_prescription_1.jpg"
            )
        )

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)

    # Set to "yolo" for YOLO segmentation + TrOCR recognition
    # Set to "text" for CRAFT line detection + TrOCR recognition + Fuzzy parsing
    # Set to "easyocr" for EasyOCR detection+recognition + Fuzzy parsing
    # Set to "yolo+easyocr" for YOLO segmentation + EasyOCR per crop
    # Set to "hybrid" for EasyOCR detection+recognition -> Parsing -> TrOCR on Rx lines
    SEGMENTATION_MODE = "hybrid"

    print(
        f"\nRunning pipeline (Segmentation Mode: {SEGMENTATION_MODE.upper()} -> TrOCR Recognition -> Postprocessing)..."
    )
    record = run_image_pipeline(image, mode=SEGMENTATION_MODE)

    print("\n--- Final Structured Record ---")
    pprint({k: v for k, v in record.items() if not k.startswith("_")})

    print("\n--- Field Confidences ---")
    pprint(record.get("_confidences", {}))

    print("\n--- Raw OCR Text (Before Lexicon Correction) ---")
    pprint(record.get("_raw_fields", {}))


if __name__ == "__main__":
    main()
