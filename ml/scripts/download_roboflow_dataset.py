import os
from roboflow import Roboflow

def download_dataset():
    """
    Downloads the Doctors' Prescriptions Handwriting dataset from Roboflow Universe.
    This script is designed to run either locally or on Kaggle.
    """
    # Initialize the Roboflow object.
    # Initialize the Roboflow object.
    # IMPORTANT: Go to https://universe.roboflow.com/
    # Search for "Doctors prescriptions handwriting"
    # Click "Download Dataset", select "YOLOv8" format (or COCO), and choose "show download code".
    # Paste that exact Python code block below:
    
    from roboflow import Roboflow
    rf = Roboflow(api_key="PASTE_YOUR_API_KEY_HERE")
    project = rf.workspace("PASTE_WORKSPACE").project("PASTE_PROJECT")
    version = project.version(1)
    dataset = version.download("yolov8")
    print("Please follow the instructions in the script to paste the Roboflow download snippet.")

    # Download complete logic handled by the snippet
    
    print(f"Downloading dataset in {download_format} format...")
    # This will download the dataset into the current directory in a folder named after the project.
    dataset = version.download(download_format)
    
    print(f"✅ Download complete! Dataset saved to: {dataset.location}")

if __name__ == "__main__":
    download_dataset()
