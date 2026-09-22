import csv
import json
import os

import cv2

import wandb
from rx_recognition_multiclass.evaluation import evaluate_batch
from rx_recognition_multiclass.pipeline import run_image_pipeline


def main():
    # Calculate paths relative to this script's directory (ml/src/rx_recognition_multiclass)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    samples_dir = os.path.join(project_root, "data", "samples")
    csv_path = os.path.join(samples_dir, "ground_truth.csv")
    results_dir = os.path.join(project_root, "ml", "results")

    if not os.path.exists(csv_path):
        print(f"Ground truth not found at {csv_path}")
        return

    # Since we are running this without a wandb account, we set it to offline
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="rx-recognition", name="baseline-tesseract")

    predictions_flat = []
    truths_flat = []

    fields = ["name", "age", "gender", "symptoms", "medicine"]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(samples_dir, row["image_file"])
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Run the baseline pipeline
            record = run_image_pipeline(image)
            raw_fields = record["_raw_fields"]

            for field in fields:
                predictions_flat.append(raw_fields.get(field, ""))
                truths_flat.append(row[field])

    # Evaluate
    results = evaluate_batch(predictions_flat, truths_flat)

    # Save locally
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "baseline.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Baseline Results:")
    print(json.dumps(results, indent=4))

    # Log to wandb (offline)
    wandb.log(
        {
            "cer": results["cer"],
            "wer": results["wer"],
            "exact_match": results["exact_match_accuracy"],
        }
    )


if __name__ == "__main__":
    main()
