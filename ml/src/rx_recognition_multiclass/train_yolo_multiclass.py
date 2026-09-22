import os

from ultralytics import YOLO


def main():
    print("Starting YOLOv8 training on the multi-class dataset...")

    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # If running on Colab, we need to download the dataset first!
    if not os.path.exists("Medical-Prescription-2"):
        print("Dataset not found locally. Downloading from Roboflow...")
        # Make sure to run: !pip install roboflow
        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key="cWkvsYfYQ9WRMi1IOCbh")
            project = rf.workspace("test-3zpx8").project("medical-prescription-3h82f")
            version = project.version(2)
            version.download("yolov8")
        except ImportError:
            print(
                "Error: roboflow library not installed. Please run '!pip install roboflow' first."
            )
            return

    yaml_path = os.path.abspath("Medical-Prescription-2/data.yaml")

    # Train YOLOv8 on the new dataset
    print(f"Training YOLOv8 on {yaml_path}...")

    # You can tweak these hyperparameters.
    # For a quick run, use 50 epochs.
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=8,
        device="cuda",  # Use GPU on Colab
        project="runs/detect",
        name="yolo_multiclass_train",
    )

    print("\nTraining complete!")
    print("The new model is saved in runs/detect/yolo_multiclass_train/weights/best.pt")


if __name__ == "__main__":
    main()
