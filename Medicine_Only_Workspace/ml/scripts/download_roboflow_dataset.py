from roboflow import Roboflow


def download_dataset():
    """
    Downloads the Doctors' Prescriptions Handwriting dataset from Roboflow Universe.
    This script is designed to run either locally or on Kaggle.
    """
    # Initialize the Roboflow object.
    # IMPORTANT: Go to https://universe.roboflow.com/
    # Go to your account settings to get your API Key.
    rf = Roboflow(api_key="cWkvsYfYQ9WRMi1IOCbh")

    # Using the dataset found at: https://universe.roboflow.com/sachin-space/doctors-prescription-new/dataset/1
    project = rf.workspace("sachin-space").project("doctors-prescription-new")
    version = project.version(1)

    print("Downloading dataset in yolov8 format...")
    # This will download the dataset into the current directory in a folder named after the project.
    dataset = version.download("yolov8")

    print(f"✅ Download complete! Dataset saved to: {dataset.location}")


if __name__ == "__main__":
    download_dataset()
