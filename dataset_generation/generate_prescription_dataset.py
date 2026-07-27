#!/usr/bin/env python3
# generate_prescription_dataset.py
"""
Synthetic prescription dataset generator for YOLOv8.

Renders synthetic prescriptions on a white 640x640 canvas with five
randomized handwritten fields (name, age, gender, symptoms, medicine) in
rotating handwriting-style Google Fonts, and writes matching YOLO-format
labels + a ready-to-use data.yaml -- no manual annotation required.

Pipeline
--------
1. Download a handful of handwriting fonts (variety) + one plain sans-serif
   font for the pre-printed form labels.
2. For each image, draw a simple prescription template, and for every
   field, draw its randomly generated value and read back the *exact*
   pixel bounding box Pillow just rendered via `ImageDraw.textbbox` --
   there is no estimated/manual box math anywhere.
3. Normalize each box to YOLO format, write one .txt label per image, and
   split everything into train/valid folders + data.yaml.

Usage:  python generate_prescription_dataset.py
Output: prescription_dataset/{data.yaml, train/{images,labels}, valid/{images,labels}}
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

# =============================================================================
# CONFIG
# =============================================================================
IMG_SIZE: int = 640
NUM_IMAGES: int = 100
TRAIN_SPLIT: float = 0.8            # 80/20 train:valid
RANDOM_SEED: int | None = None      # set an int for a reproducible dataset

OUTPUT_DIR = Path("prescription_dataset")
FONT_DIR = Path("fonts")

CLASSES: list[str] = ["name", "age", "gender", "symptoms", "medicine"]
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# Verified static (non-variable) handwriting Google Fonts. Several styles
# rotate randomly per image so the detector learns "handwriting" in general
# rather than memorizing one specific font.
CURSIVE_FONT_URLS: dict[str, str] = {
    "Pacifico": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "ShadowsIntoLight": "https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "ReenieBeanie": "https://raw.githubusercontent.com/google/fonts/main/ofl/reeniebeanie/ReenieBeanie.ttf",
    "IndieFlower": "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "Kalam": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf",
    "PatrickHand": "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
}
FALLBACK_LABEL_FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/apache/roboto/static/Roboto-Regular.ttf"

INK_COLORS = ["#000000", "#0b2e6b", "#1a1a2e", "#26261f"]

CLINIC_NAMES = [
    "City Care Clinic", "Sunrise Medical Center", "Green Valley Hospital",
    "Wellness Family Clinic", "St. Mary's Health Center", "Metro General Hospital",
    "Lakeside Multispeciality Clinic", "Apex Diagnostic & Care",
]
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Ishaan", "Krishna", "Rohan",
    "Priya", "Ananya", "Diya", "Saanvi", "Kiara", "Anika", "Meera", "Riya",
    "John", "Michael", "David", "James", "Robert", "William", "Daniel",
    "Mary", "Patricia", "Linda", "Barbara", "Susan", "Emma", "Olivia",
]
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Kumar", "Singh", "Patel", "Reddy", "Rao",
    "Nair", "Iyer", "Menon", "Smith", "Johnson", "Williams", "Brown",
    "Jones", "Garcia", "Miller", "Davis", "Wilson",
]
SYMPTOMS_LIST = [
    "Fever and dry cough", "Severe headache", "Body ache and fatigue",
    "Sore throat, mild fever", "Nausea and vomiting", "Chest congestion",
    "Joint pain and swelling", "Skin rash with itching", "Abdominal pain",
    "Dizziness and weakness", "Cold, cough and sneezing",
    "Shortness of breath", "Loss of appetite", "Mild fever since 3 days",
]
MEDICINE_LIST = [
    "Tab. Paracetamol 500mg 1-0-1", "Tab. Azithromycin 500mg 1-0-0",
    "Cap. Amoxicillin 250mg 1-1-1", "Tab. Cetirizine 10mg 0-0-1",
    "Syp. Ambroxol 5ml 1-1-1", "Tab. Ibuprofen 400mg 1-0-1",
    "Tab. Omeprazole 20mg 1-0-0", "Tab. Metformin 500mg 1-0-1",
    "Cap. Doxycycline 100mg 1-0-1", "Tab. Vitamin D3 60000IU Weekly",
    "Tab. Montelukast 10mg 0-0-1", "Syp. Cough Syrup 10ml 1-1-1",
]


# =============================================================================
# FONT SETUP
# =============================================================================
def _download(url: str, dest: Path, timeout: int = 15) -> None:
    """Downloads `url` to `dest`, skipping if already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        dest.write_bytes(resp.read())


def setup_cursive_fonts() -> list[Path]:
    """Downloads the handwriting fonts, skipping any that fail, and returns
    the paths that loaded successfully as real fonts."""
    good: list[Path] = []
    for name, url in CURSIVE_FONT_URLS.items():
        dest = FONT_DIR / f"{name}.ttf"
        try:
            _download(url, dest)
            ImageFont.truetype(str(dest), 20)  # sanity check it's a real font
            good.append(dest)
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
    if not good:
        raise RuntimeError(
            "Could not download any handwriting fonts. If you're on Kaggle, "
            "enable internet in Notebook Settings, or upload .ttf files "
            "yourself and point CURSIVE_FONT_URLS/FONT_DIR at them."
        )
    print(f"Using {len(good)} handwriting fonts: {[f.stem for f in good]}")
    return good


def get_label_font_path() -> Path:
    """Finds a plain sans-serif font for the pre-printed form labels. Tries
    matplotlib's bundled DejaVu Sans first (present on virtually every
    Kaggle/Colab image, no internet needed), then common system paths, then
    downloads a static Google Font as a last resort."""
    try:
        import matplotlib
        mpl_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for candidate in (mpl_ttf / "DejaVuSans-Bold.ttf", mpl_ttf / "DejaVuSans.ttf"):
            if candidate.exists():
                return candidate
    except ImportError:
        pass

    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.exists():
            return candidate

    dest = FONT_DIR / "Roboto-Regular.ttf"
    _download(FALLBACK_LABEL_FONT_URL, dest)
    return dest


# =============================================================================
# RANDOM CONTENT
# =============================================================================
def random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_age() -> str:
    return str(random.randint(1, 92))


def random_gender() -> str:
    return random.choice(["Male", "Female"])


def random_symptoms() -> str:
    return random.choice(SYMPTOMS_LIST)


def random_medicines() -> list[str]:
    return random.sample(MEDICINE_LIST, random.randint(1, 2))


# =============================================================================
# DRAWING + EXACT BBOX HELPERS
# =============================================================================
def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word-wrap using real measured pixel widths (not char counts)."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_field(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value_lines: Sequence[str],
    label_font: FreeTypeFont,
    value_font: FreeTypeFont,
    class_id: int,
    annotations: list[tuple[int, tuple[float, float, float, float]]],
    ink: str,
    block: bool = False,
    indent: int = 22,
) -> int:
    """Draws a printed `label` plus a handwritten value, and records ONE
    ground-truth box for the *value only* -- the printed label is template
    decoration, not one of the five classes we want the detector to find.

    `draw.textbbox` returns the exact (left, top, right, bottom) box Pillow
    is about to render for that text/font/size, so the saved box is exactly
    what ends up on the image -- never an estimate.

    block=False: value drawn inline, right of the label, same row.
    block=True : value drawn on the line(s) below the label (for multi-line
                 fields like symptoms/medicine).
    Returns the y-coordinate of the bottom of everything just drawn.
    """
    draw.text((x, y), label, font=label_font, fill="#333333")
    label_bbox = draw.textbbox((x, y), label, font=label_font)

    if not block:
        vx, vy = label_bbox[2] + 14, y - 3
        value = value_lines[0]
        vbbox = draw.textbbox((vx, vy), value, font=value_font)
        draw.text((vx, vy), value, font=value_font, fill=ink)
        annotations.append((class_id, vbbox))
        return max(label_bbox[3], vbbox[3])

    vy = label_bbox[3] + 8
    vx = x + indent
    boxes = []
    for line in value_lines:
        b = draw.textbbox((vx, vy), line, font=value_font)
        draw.text((vx, vy), line, font=value_font, fill=ink)
        boxes.append(b)
        vy = b[3] + 6
    merged = (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )
    annotations.append((class_id, merged))
    return merged[3]


# =============================================================================
# ONE PRESCRIPTION IMAGE
# =============================================================================
def generate_prescription(
    cursive_fonts: Sequence[Path], label_font_path: Path
) -> tuple[Image.Image, list[tuple[int, tuple[float, float, float, float]]]]:
    """Renders one synthetic prescription. Returns (image, annotations),
    where annotations is [(class_id, pixel_bbox), ...]."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    annotations: list[tuple[int, tuple[float, float, float, float]]] = []

    ink = random.choice(INK_COLORS)
    value_font = ImageFont.truetype(str(random.choice(cursive_fonts)), random.randint(25, 30))
    label_font = ImageFont.truetype(str(label_font_path), 19)
    title_font = ImageFont.truetype(str(label_font_path), 24)

    margin = 42
    y = 32

    # --- decorative header (template chrome, not an annotated class) ---
    clinic = random.choice(CLINIC_NAMES)
    draw.text((margin, y), clinic, font=title_font, fill="#111111")
    y = draw.textbbox((margin, y), clinic, font=title_font)[3] + 10
    draw.line([(margin, y), (IMG_SIZE - margin, y)], fill="#999999", width=2)
    y += 24

    # --- name (inline) ---
    y = draw_field(draw, margin, y, "Patient Name:", [random_name()],
                    label_font, value_font, CLASS_TO_ID["name"], annotations, ink) + 26

    # --- age + gender, same row (inline) ---
    row_y = y
    y1 = draw_field(draw, margin, row_y, "Age:", [random_age()],
                     label_font, value_font, CLASS_TO_ID["age"], annotations, ink)
    y2 = draw_field(draw, margin + 300, row_y, "Gender:", [random_gender()],
                     label_font, value_font, CLASS_TO_ID["gender"], annotations, ink)
    y = max(y1, y2) + 28

    # --- symptoms (block, wraps to fit canvas width) ---
    max_w = IMG_SIZE - margin - 22 - margin
    symptom_lines = wrap_to_width(draw, random_symptoms(), value_font, max_w)[:2]
    y = draw_field(draw, margin, y, "Symptoms:", symptom_lines,
                    label_font, value_font, CLASS_TO_ID["symptoms"], annotations,
                    ink, block=True) + 28

    # --- medicine (block, 1-2 meds, wraps to fit canvas width) ---
    med_lines: list[str] = []
    for med in random_medicines():
        med_lines.extend(wrap_to_width(draw, med, value_font, max_w))
    med_lines = med_lines[:3]
    y = draw_field(draw, margin, y, "Medicine:", med_lines,
                    label_font, value_font, CLASS_TO_ID["medicine"], annotations,
                    ink, block=True) + 30

    # --- decorative footer (template chrome, not annotated) ---
    draw.line([(IMG_SIZE - 230, y + 46), (IMG_SIZE - margin, y + 46)], fill="#999999", width=1)
    draw.text((IMG_SIZE - 220, y + 52), "Doctor's Signature", font=label_font, fill="#777777")

    return img, annotations


# =============================================================================
# YOLO OUTPUT
# =============================================================================
def to_yolo_line(class_id: int, bbox: tuple[float, float, float, float]) -> str:
    left, top, right, bottom = bbox
    left = max(0.0, min(IMG_SIZE, left))
    top = max(0.0, min(IMG_SIZE, top))
    right = max(0.0, min(IMG_SIZE, right))
    bottom = max(0.0, min(IMG_SIZE, bottom))
    xc = (left + right) / 2 / IMG_SIZE
    yc = (top + bottom) / 2 / IMG_SIZE
    w = (right - left) / IMG_SIZE
    h = (bottom - top) / IMG_SIZE
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def save_sample(
    img: Image.Image,
    annotations: list[tuple[int, tuple[float, float, float, float]]],
    index: int,
    split: str,
) -> bool:
    """Saves image + label file. Returns True if any box came close to the
    bottom edge (used as an overflow warning during generation)."""
    img_path = OUTPUT_DIR / split / "images" / f"prescription_{index:04d}.jpg"
    lbl_path = OUTPUT_DIR / split / "labels" / f"prescription_{index:04d}.txt"
    img.save(img_path, quality=95)
    lbl_path.write_text("\n".join(to_yolo_line(c, b) for c, b in annotations) + "\n")
    return any(b[3] > IMG_SIZE - 6 for _, b in annotations)


def write_data_yaml() -> None:
    lines = [
        "# Auto-generated by generate_prescription_dataset.py",
        "# 'train'/'val' below are relative to this file's folder. If your",
        "# working directory differs, uncomment 'path' and set it explicitly:",
        "# path: /kaggle/working/prescription_dataset",
        "",
        "train: train/images",
        "val: valid/images",
        "",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    lines += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (OUTPUT_DIR / "data.yaml").write_text("\n".join(lines) + "\n")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    for split in ("train", "valid"):
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    print("Setting up fonts...")
    cursive_fonts = setup_cursive_fonts()
    label_font_path = get_label_font_path()
    print(f"Label font: {label_font_path}")

    order = list(range(NUM_IMAGES))
    random.shuffle(order)
    n_train = int(NUM_IMAGES * TRAIN_SPLIT)
    train_ids = set(order[:n_train])

    overflow = 0
    for i in range(NUM_IMAGES):
        split = "train" if i in train_ids else "valid"
        img, annotations = generate_prescription(cursive_fonts, label_font_path)
        if save_sample(img, annotations, i, split):
            overflow += 1

    write_data_yaml()

    print(f"\nGenerated {n_train} train / {NUM_IMAGES - n_train} valid images -> {OUTPUT_DIR}/")
    if overflow:
        print(f"Note: {overflow} image(s) had a field near the bottom edge; "
              f"lower the value_font random.randint() range in generate_prescription() "
              f"if you see clipped text.")


if __name__ == "__main__":
    main()
