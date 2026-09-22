import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

import cv2

load_dotenv()  # Load environment variables from .env file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "src")))

# Import the existing heuristic pipeline unmodified
from rx_recognition_heuristic.pipeline import run_heuristic_pipeline

# Import the new LLM advice module
from rx_llm_advice.dosage_advice import generate_dosage_advice


def main():
    parser = argparse.ArgumentParser(
        description="Run the Heuristic Pipeline + LLM Dosage Advice"
    )
    parser.add_argument("image", help="Path to the prescription image")
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image {image_path}")
        sys.exit(1)

    # Extract base name for the output folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Saving outputs to directory: {output_dir}")

    print(f"\n[1] Running Heuristic Pipeline on: {image_path}")
    print("--------------------------------------------------")
    # Run the unmodified pipeline, now saving debug images to output_dir
    record = run_heuristic_pipeline(image, output_dir=output_dir)

    # Save normal (heuristic) output
    normal_out_file = os.path.join(output_dir, "output_normal.json")
    with open(normal_out_file, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[INFO] Saved normal pipeline output to {normal_out_file}")

    print("\n[2] Extracting Medicines and Querying LLM...")
    print("--------------------------------------------------")
    # Extract medicines from the record
    medicines = record.get("Medical Text", [])

    if isinstance(medicines, str):
        medicines = medicines.split(" | ")

    if medicines:
        advice = generate_dosage_advice(medicines)
    else:
        advice = "No medicines detected in the prescription."

    print(f"\n[Dosage Advice via Gemini]\n{advice}\n")

    # Append the advice to the JSON record
    record["Dosage Advice"] = advice

    # Save and output the final LLM-augmented JSON
    llm_out_file = os.path.join(output_dir, "output_llm.json")
    with open(llm_out_file, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[INFO] Saved LLM-augmented output to {llm_out_file}")

    print("\n--- Final Structured Record ---")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
