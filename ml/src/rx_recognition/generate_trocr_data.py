import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def generate_dummy_data(output_dir="data/trocr-dataset"):
    print(f"Generating dummy TrOCR data in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # We will generate 5 tiny images with dummy text
    records = []

    words = ["Paracetamol", "Amoxicillin", "500mg", "Twice daily", "Ibuprofen"]

    # Try to load a basic font, or fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        font = ImageFont.load_default()

    for i, word in enumerate(words):
        img_path = os.path.join(output_dir, f"dummy_{i}.jpg")

        # Create a white background image (roughly the size of a cropped word/field)
        img = Image.new("RGB", (300, 100), color="white")
        draw = ImageDraw.Draw(img)

        # Draw the text
        draw.text((20, 30), word, fill="black", font=font)

        # Save image
        img.save(img_path)

        # Record in manifest
        records.append(
            {
                "image_path": img_path,
                "field_type": "medicine_name" if i in [0, 1, 4] else "dosage",
                "transcription": word,
            }
        )

    # Create manifest
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "train_manifest.csv")
    df.to_csv(csv_path, index=False)

    # For the smoke test, we'll just use the same 5 images for validation too
    val_csv_path = os.path.join(output_dir, "val_manifest.csv")
    df.to_csv(val_csv_path, index=False)

    print(f"Successfully generated 5 images. Manifests saved to {csv_path} and {val_csv_path}.")


if __name__ == "__main__":
    generate_dummy_data()
