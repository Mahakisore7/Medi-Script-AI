import os
import shutil

from ultralytics import YOLO


def train():
    """
    Train YOLOv8n on the generated prescription dataset (100 images).
    This trains a model to detect 5 fields: name, age, gender, symptoms, medicine.
    """
    print("Starting YOLOv8n training on generated prescription dataset...")

    # Load the pretrained nano model (downloads ~6MB if not cached)
    model = YOLO("yolov8n.pt")

    # Point to the generated dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys

    from mediscript.config import DATASETS_DIR

    yaml_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(DATASETS_DIR / "doctors-prescription-new" / "data.yaml")
    )

    if not os.path.exists(yaml_path):
        print(f"Error: Could not find {yaml_path}")
        print("Make sure you generated the dataset first! See ml/src/rx_generation/README.md")
        return

    # Train YOLOv8 on our synthetic bounding boxes
    print(f"Training YOLOv8 on {yaml_path}...")
    model.train(
        data=yaml_path,
        epochs=30,  # Real dataset, 30 epochs is good for 200 images
        imgsz=640,  # Standard YOLO input size
        batch=8,  # Reasonable batch for CPU/small GPU
        project="runs/detect",
        name="prescription_fields",
        patience=10,  # Early stopping if no improvement for 10 epochs
        save=True,
        exist_ok=True,
    )

    # Copy best weights to the standard location the pipeline expects
    best_weights = os.path.join("runs", "detect", "prescription_fields", "weights", "best.pt")
    target_path = os.path.join("runs", "detect", "kaggle_best.pt")

    if os.path.exists(best_weights):
        shutil.copy2(best_weights, target_path)
        print(f"\n✅ Training complete! Best weights copied to: {target_path}")
    else:
        print(f"\n⚠️ Training finished but best.pt not found at {best_weights}")

    print("You can now run demo.py with SEGMENTATION_MODE='yolo+easyocr'")


if __name__ == "__main__":
    train()
