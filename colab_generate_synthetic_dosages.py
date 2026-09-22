"""
==========================================================================
  Medi-Script-AI: Synthetic Dosage Data Generator (Google Colab)
  -----------------------------------------------------------------------
  Run this script in a new cell in your Colab notebook (after mounting Google Drive)
  to automatically generate 1000 synthetic handwriting examples of "1-0-1 x 5 days"
  and append them to your Roboflow_TrOCR_Dataset!
==========================================================================
"""

import os
import random
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import uuid

# 1. Configuration
# Make sure this points to the exact folder where your images/ and labels.csv are!
DATASET_DIR = "/content/drive/MyDrive/Medi-Script-AI/Roboflow_TrOCR_Dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

NUM_IMAGES_TO_GENERATE = 1000

# 2. Download Fonts
FONTS = {
    "Pacifico": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "PermanentMarker": "https://raw.githubusercontent.com/google/fonts/main/apache/permanentmarker/PermanentMarker-Regular.ttf",
    "AmaticSC": "https://raw.githubusercontent.com/google/fonts/main/ofl/amaticsc/AmaticSC-Regular.ttf",
    "Sacramento": "https://raw.githubusercontent.com/google/fonts/main/ofl/sacramento/Sacramento-Regular.ttf",
}

os.makedirs("fonts", exist_ok=True)
font_paths = []
print("Downloading cursive fonts...")
for name, url in FONTS.items():
    font_path = f"fonts/{name}.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(url, font_path)
    font_paths.append(font_path)
print("Fonts ready.")


# 3. Dosage Generator Logic
def generate_random_dosage_text():
    # Parts of a typical Indian prescription dosage
    frequencies = [
        "1-0-1",
        "1 - 0 - 1",
        "1-1-1",
        "1 - 1 - 1",
        "0-1-0",
        "0 - 1 - 0",
        "1-0-0",
        "1 - 0 - 0",
        "0-0-1",
        "0 - 0 - 1",
        "1-1-1-1",
        "2-0-2",
        "1/2 - 0 - 1/2",
    ]
    durations = [
        "x 5 days",
        "x5 days",
        "x 5 Days",
        "x 5days",
        "x 3 days",
        "x 7 days",
        "x 10 days",
        "x 15 days",
        "x 1 week",
        "x 2 weeks",
        "x 1 month",
        "for 5 days",
        "for 1 week",
    ]

    freq = random.choice(frequencies)
    if random.random() > 0.3:  # 70% chance to have a duration
        dur = random.choice(durations)
        return f"{freq} {dur}"
    return freq


# 4. Image Generation
def create_synthetic_image(text, font_path):
    # Randomize font size and image dimensions slightly
    font_size = random.randint(40, 70)
    font = ImageFont.truetype(font_path, font_size)

    # Get text bounding box
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Add padding
    img_width = text_width + random.randint(20, 60)
    img_height = text_height + random.randint(20, 60)

    # Create image with off-white background
    bg_color = (
        random.randint(240, 255),
        random.randint(240, 255),
        random.randint(240, 255),
    )
    image = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(image)

    # Text color (usually black or dark blue pen)
    pen_color = random.choice([(0, 0, 0), (10, 10, 50), (20, 20, 80)])

    # Draw text
    x_offset = random.randint(10, 30)
    y_offset = random.randint(10, 30)
    draw.text((x_offset, y_offset), text, font=font, fill=pen_color)

    # Augmentations (Rotation)
    if random.random() > 0.5:
        angle = random.uniform(-3, 3)
        image = image.rotate(
            angle, resample=Image.BICUBIC, expand=True, fillcolor=bg_color
        )

    # Augmentations (Blur/Noise)
    if random.random() > 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    return image


# 5. Main Execution
if not os.path.exists("/content/drive/MyDrive"):
    print("CRITICAL ERROR: Google Drive is not mounted!")
    print(
        "Please run this code in a cell first:\nfrom google.colab import drive\ndrive.mount('/content/drive')"
    )
else:
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset folder not found at {DATASET_DIR}. Creating it now...")
        os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"Generating {NUM_IMAGES_TO_GENERATE} synthetic dosage images...")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    new_records = []

    for i in range(NUM_IMAGES_TO_GENERATE):
        text = generate_random_dosage_text()
        font_path = random.choice(font_paths)

        img = create_synthetic_image(text, font_path)

        # Save image
        filename = f"synth_dosage_{uuid.uuid4().hex[:8]}.jpg"
        img.save(os.path.join(IMAGES_DIR, filename), quality=random.randint(85, 95))

        # Add to records
        new_records.append({"Images": filename, "Text": text})

        if (i + 1) % 200 == 0:
            print(f"Generated {i+1}/{NUM_IMAGES_TO_GENERATE}...")

    print("Finished generating images. Appending to labels.csv...")

    # Append to labels.csv
    df_new = pd.DataFrame(new_records)

    if os.path.exists(LABELS_CSV):
        df_existing = pd.read_csv(LABELS_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print(f"WARNING: labels.csv not found at {LABELS_CSV}. Creating a new one.")
        df_combined = df_new

    df_combined.to_csv(LABELS_CSV, index=False)

    print(f"\nSUCCESS! Added {NUM_IMAGES_TO_GENERATE} dosage examples to your dataset.")
    print("You can now run your TrOCR training cell!")
