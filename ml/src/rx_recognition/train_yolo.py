import os

from ultralytics import YOLO


def train():
    """
    Local smoke test for YOLOv8n fine-tuning.
    This uses a tiny batch and 1 epoch just to prove the pipeline works locally
    before you take it to Kaggle for the real 60-epoch training run.
    """
    print("Starting YOLOv8n smoke test...")

    # Load the pretrained nano model (downloads ~6MB if not cached)
    model = YOLO("yolov8n.pt")

    # We point data to our generated YAML file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(script_dir, "..", "..", "rx_fields.yaml"))

    # Run a tiny training session
    model.train(
        data=yaml_path,
        epochs=1,  # Just 1 epoch for smoke test
        imgsz=416,  # Small image size for speed
        batch=2,  # Tiny batch
        project="rx-recognition",
        name="yolov8n-smoke-test",
    )

    print("Smoke test complete!")
    print("Your weights are saved in: runs/detect/yolov8n-smoke-test/weights/best.pt")


if __name__ == "__main__":
    train()
