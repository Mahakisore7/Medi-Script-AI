import cv2
import os
from pprint import pprint
from rx_recognition.pipeline import run_image_pipeline


def main():
    # Load a sample image
    image_path = "data/samples/synthetic_sample.jpg"
    print(f"Loading image from {image_path}...")

    if not os.path.exists(image_path):
        print(f"Error: Could not find {image_path}")
        return

    image = cv2.imread(image_path)

    print(
        "\nRunning pipeline (YOLO Segmentation -> TrOCR Recognition -> Postprocessing)..."
    )
    record = run_image_pipeline(image)

    print("\n--- Final Structured Record ---")
    pprint({k: v for k, v in record.items() if not k.startswith("_")})

    print("\n--- Field Confidences ---")
    pprint(record.get("_confidences", {}))

    print("\n--- Raw OCR Text (Before Lexicon Correction) ---")
    pprint(record.get("_raw_fields", {}))


if __name__ == "__main__":
    main()
