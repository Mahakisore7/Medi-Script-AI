import os
import logging
import cv2
from pprint import pprint
from ml.src.rx_recognition.pipeline import run_image_pipeline

# Show INFO logs so we can see LocateAnything output clearly
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    # Load a sample image
    # Switch between a real prescription and a test image
    image_path = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\my sample\original_prescription_1.jpg"

    print(f"Loading image from {image_path}...")

    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}")
        return

    image = cv2.imread(image_path)

    # Set to "yolo" for fast, reliable extraction (Current Progress)
    # Set to "locateanything" for experimental zero-shot extraction via local API
    SEGMENTATION_MODE = "yolo"


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
