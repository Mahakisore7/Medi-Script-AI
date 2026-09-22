from roboflow import Roboflow


def main():
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="cWkvsYfYQ9WRMi1IOCbh")
    project = rf.workspace("test-3zpx8").project("medical-prescription-3h82f")
    version = project.version(2)
    # The download method automatically extracts the dataset to a folder named after the project
    dataset = version.download("yolov8")

    print(f"Dataset downloaded successfully to: {dataset.location}")


if __name__ == "__main__":
    main()
