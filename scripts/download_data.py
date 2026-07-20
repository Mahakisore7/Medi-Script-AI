import os
import subprocess

def download_kaggle_dataset(dataset_slug, download_dir):
    """
    Downloads and unzips a Kaggle dataset using the Kaggle CLI.
    Requires kaggle.json to be configured in ~/.kaggle/
    """
    print(f"Downloading {dataset_slug} to {download_dir}...")
    os.makedirs(download_dir, exist_ok=True)
    
    # Download
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset_slug, "-p", download_dir
    ], check=True)
    
    # Unzip
    zip_filename = dataset_slug.split("/")[-1] + ".zip"
    zip_filepath = os.path.join(download_dir, zip_filename)
    
    print(f"Unzipping {zip_filepath}...")
    subprocess.run(["unzip", "-q", "-o", zip_filepath, "-d", download_dir], check=True)
    
    # Clean up zip file
    os.remove(zip_filepath)
    print(f"Finished downloading and extracting {dataset_slug}")

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    
    # 1. IAM Dataset (Kaggle mirror for pretraining)
    iam_dataset = "nibinv23/iam-handwriting-word-database"
    download_kaggle_dataset(iam_dataset, os.path.join(DATA_DIR, "iam_dataset"))
    
    # 2. Doctor's Handwritten Prescription BD Dataset (for fine-tuning later)
    # The exact slug might vary, but this is a common placeholder for it.
    # Update this slug if the exact Kaggle dataset URL is different.
    bd_dataset = "samyaksaha/doctors-handwritten-prescription-bd-dataset" 
    download_kaggle_dataset(bd_dataset, os.path.join(DATA_DIR, "bd_prescription_dataset"))
    
    print("All datasets downloaded successfully!")
