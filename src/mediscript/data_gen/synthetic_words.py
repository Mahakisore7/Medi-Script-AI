import os
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from generate_prescription_dataset import setup_cursive_fonts

# Configuration
NUM_STYLES_PER_WORD = 3
IMG_HEIGHT = 64  # Standard TrOCR image height
PADDING = 10
OUTPUT_DIR = Path("synthetic_trocr_data")

def generate_word_image(text, font, augment=True):
    # Estimate size
    dummy_img = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 2 * PADDING
    height = max(IMG_HEIGHT, bbox[3] - bbox[1] + 2 * PADDING)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # Random text color (black/dark blue/grey)
    color = random.choice(["#000000", "#111111", "#0b2e6b", "#1a1a2e"])
    
    # Draw text centered vertically
    y_offset = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((PADDING, y_offset), text, font=font, fill=color)

    if augment:
        # Rotation
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
        
        # Blur
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))

    # Resize height to IMG_HEIGHT while maintaining aspect ratio, as required by TrOCR
    new_width = max(1, int(img.width * (IMG_HEIGHT / img.height)))
    img = img.resize((new_width, IMG_HEIGHT), Image.Resampling.LANCZOS)
    return img

def process_lexicon(csv_path, field_type, fonts, split="train"):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    records = []
    out_dir = OUTPUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, word in enumerate(words):
        if not word:
            continue
        for style_idx in range(NUM_STYLES_PER_WORD):
            font_path = random.choice(fonts)
            font_size = random.randint(32, 48)
            font = ImageFont.truetype(str(font_path), font_size)
            
            img = generate_word_image(word, font)
            
            # Create a safe filename limit length
            safe_word = "".join([c if c.isalnum() else "_" for c in word])[:20]
            filename = f"{field_type}_{safe_word}_{i}_{style_idx}.jpg"
            filepath = out_dir / filename
            img.save(filepath, quality=random.randint(85, 95))
            
            records.append({
                "image_path": str(filepath.relative_to(OUTPUT_DIR.parent)),
                "field_type": field_type,
                "transcription": word
            })
            
    return records

def main():
    print("Setting up fonts...")
    fonts = setup_cursive_fonts()
    
    project_root = Path(__file__).parent.parent
    lexicon_dir = project_root / "data" / "lexicon"
    
    records = []
    
    # Medicines
    print("Generating synthetic medicines...")
    records.extend(process_lexicon(lexicon_dir / "medicines.csv", "medicine", fonts))
    
    # Symptoms
    print("Generating synthetic symptoms...")
    records.extend(process_lexicon(lexicon_dir / "symptoms.csv", "symptoms", fonts))
    
    if records:
        df = pd.DataFrame(records)
        manifest_path = OUTPUT_DIR / "synthetic_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"Generated {len(records)} synthetic images. Manifest saved at {manifest_path}")

if __name__ == "__main__":
    main()
