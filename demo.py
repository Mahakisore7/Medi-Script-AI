import os
import sys
import argparse
import cv2
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "src")))

from rx_recognition.pipeline import run_image_pipeline
from rx_recognition_heuristic.pipeline import run_heuristic_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Medi-Script-AI pipeline")
    parser.add_argument(
        "image",
        nargs="?",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "my sample",
                "original_prescription_1.jpg",
            )
        ),
        help="Path to the image",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Path to the YOLO ground truth label (.txt). Guarantees perfect TrOCR crops for testing.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="heuristic",
        choices=["paddleocr", "yolo", "hybrid", "paddle_hybrid", "text", "heuristic"],
        help="Segmentation mode. Use 'heuristic' for perfect hybrid behavior.",
    )
    args = parser.parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image with OpenCV")
        return

    SEGMENTATION_MODE = args.mode

    if SEGMENTATION_MODE == "heuristic":
        print(f"\nRunning HEURISTIC pipeline (Label Path: {args.label})...")
        record = run_heuristic_pipeline(image, label_path=args.label)
        print("\n--- Final Structured Record ---")
        import json

        print(json.dumps(record, indent=2))
    else:
        print(
            f"\nRunning pipeline (Segmentation Mode: {SEGMENTATION_MODE.upper()} -> TrOCR Recognition -> Postprocessing)..."
        )
        record = run_image_pipeline(image, mode=SEGMENTATION_MODE)
        print("\n--- Final Structured Record ---")
        pprint({k: v for k, v in record.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
