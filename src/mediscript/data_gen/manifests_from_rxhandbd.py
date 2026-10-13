import os

import pandas as pd


def generate_trocr_manifests(dataset_dir):
    print(f"Processing TrOCR dataset in {dataset_dir}...")

    # Paths for training data
    train_csv = os.path.join(dataset_dir, "Training", "training_labels.csv")
    train_img_dir = os.path.join(dataset_dir, "Training", "training_words")

    # Paths for validation data
    val_csv = os.path.join(dataset_dir, "Validation", "validation_labels.csv")
    val_img_dir = os.path.join(dataset_dir, "Validation", "validation_words")

    output_dir = os.path.dirname(dataset_dir)  # data/trocr-dataset

    # Function to process a split
    def process_split(csv_path, img_dir, split_name):
        if not os.path.exists(csv_path):
            print(f"ERROR: Could not find {csv_path}")
            return None

        print(f"Processing {split_name} split...")
        df = pd.read_csv(csv_path)

        records = []
        for _, row in df.iterrows():
            img_filename = str(row["IMAGE"])
            img_path_abs = os.path.abspath(os.path.join(img_dir, img_filename))
            # Save relative path so it works when uploaded to Kaggle
            img_path = os.path.relpath(img_path_abs, output_dir)

            # The exact ground truth text
            transcription = str(row["MEDICINE_NAME"])

            if os.path.exists(img_path_abs):
                records.append(
                    {
                        "image_path": img_path,
                        "field_type": "medicine",
                        "transcription": transcription,
                    }
                )
            else:
                print(f"Warning: Image {img_path_abs} not found.")

        # Create manifest
        manifest_df = pd.DataFrame(records)
        out_csv = os.path.join(output_dir, f"{split_name}_manifest.csv")
        manifest_df.to_csv(out_csv, index=False)
        print(f"Saved {len(records)} records to {out_csv}")
        return manifest_df

    process_split(train_csv, train_img_dir, "train")
    process_split(val_csv, val_img_dir, "val")
    print("Done generating TrOCR manifests!")


if __name__ == "__main__":
    # Point to the Kaggle dataset extracted folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    dataset_folder = os.path.join(
        project_root, "data", "trocr-dataset", "Doctor’s Handwritten Prescription BD dataset"
    )

    generate_trocr_manifests(dataset_folder)
