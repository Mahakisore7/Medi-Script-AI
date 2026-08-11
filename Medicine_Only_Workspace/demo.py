import os
import logging
import sys
import cv2
from pprint import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "src")))

from rx_recognition.pipeline import run_image_pipeline

# Show INFO logs so we can see LocateAnything output clearly
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Medi-Script-AI pipeline")
    parser.add_argument("image", nargs="?", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "my sample", "original_prescription_1.jpg")), help="Path to the image")
    parser.add_argument("--mode", type=str, default="paddleocr", choices=["paddleocr", "yolo", "hybrid", "paddle_hybrid", "text"], help="Segmentation mode")
    args = parser.parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)
    SEGMENTATION_MODE = args.mode

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

    # Display the debug annotated image
    debug_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "runs", "debug", "segmentation_debug.jpg"))
    if os.path.exists(debug_path):
        print(f"\nOpening debug image: {debug_path}")
        debug_img = cv2.imread(debug_path)
        # Resize if it's too big to fit on a standard laptop screen
        h, w = debug_img.shape[:2]
        if h > 800:
            scale = 800 / h
            debug_img = cv2.resize(debug_img, (int(w * scale), int(h * scale)))
        
        cv2.imshow("Annotated Image - Press any key to close", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
