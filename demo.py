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
    image_path = "data/samples/human_sample_01.jpeg"

    print(f"Loading image from {image_path}...")

    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}")
        return

    image = cv2.imread(image_path)

    # Set to "yolo" for the original field detector
    # Set to "dbnet" for text-region detection followed by field mapping
    # Set to "locateanything" for the experimental zero-shot local API mode
    SEGMENTATION_MODE = os.getenv("SEGMENTATION_MODE", "yolo").lower()
    DBNET_MODEL_PATH = os.getenv("DBNET_MODEL_PATH")

    print(
        f"\nRunning pipeline (Segmentation Mode: {SEGMENTATION_MODE.upper()} -> TrOCR Recognition -> Postprocessing)..."
    )
    record = run_image_pipeline(
        image,
        mode=SEGMENTATION_MODE,
        dbnet_model_path=DBNET_MODEL_PATH,
    )

    print("\n--- Final Structured Record ---")
    pprint({k: v for k, v in record.items() if not k.startswith("_")})

    print("\n--- Field Confidences ---")
    pprint(record.get("_confidences", {}))

    print("\n--- Raw OCR Text (Before Lexicon Correction) ---")
    pprint(record.get("_raw_fields", {}))

    print("\n--- Debug Artifacts ---")
    print(record.get("_debug_dir", "No debug directory created"))


if __name__ == "__main__":
    main()
