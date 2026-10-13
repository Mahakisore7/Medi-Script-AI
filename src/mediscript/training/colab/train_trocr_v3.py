"""
==========================================================================
  Medi-Script-AI: TrOCR Fine-Tuning Script v3 — Cursive Specialization
  -----------------------------------------------------------------------
  KEY UPGRADES OVER v2:
    1. Unfrozen top 4 encoder layers — model actually LEARNS to see cursive
    2. Elastic + Perspective augmentations — trains on every slant/stretch
    3. Medical vocabulary injection — fewer tokenizer sub-words per drug
    4. Cosine LR scheduler with warmup — gradual, safer encoder fine-tuning
    5. Lower LR (5e-6) — avoids catastrophic forgetting of English knowledge
    6. More epochs (20) with patience=5 early stopping

  Usage: Run each cell in order in Google Colab (GPU runtime T4 or better).
  IMPORTANT: Your dataset folder structure must be:
    /content/drive/MyDrive/Medi-Script-AI/Roboflow_TrOCR_Dataset/
      images/         <- crop images (PNG or JPG)
      labels.csv      <- columns: "Images", "Text"
==========================================================================
"""

# ===================== CELL 1: Install Dependencies =====================
# Run this cell first!
# !pip install -q transformers[torch] datasets evaluate jiwer pillow pandas scikit-learn opencv-python-headless

# ===================== CELL 2: Mount Google Drive =======================
# from google.colab import drive
# drive.mount('/content/drive')

import os
import random

import cv2
import evaluate
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    RobertaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

# >>> CHANGE THIS PATH to wherever you uploaded the dataset in Drive <<<
DATASET_DIR = "/content/drive/MyDrive/Medi-Script-AI/Roboflow_TrOCR_Dataset"

# Training Hyperparameters
# V3: Much lower LR (was 5e-5). Unfreezing the encoder requires a gentler
# learning rate to avoid catastrophic forgetting.
EPOCHS = 20
BATCH_SIZE = 4  # Reduce to 2 if you get CUDA OOM errors on T4
LEARNING_RATE = 5e-6  # 10x lower than v2 - critical for safe encoder training
MAX_TOKEN_LENGTH = 128  # Increased from 64 - some prescriptions are long
TRAIN_SPLIT = 0.85

OUTPUT_DIR = "/content/drive/MyDrive/Medi-Script-AI/trocr_v3_training_output"
FINAL_MODEL_DIR = "/content/drive/MyDrive/Medi-Script-AI/final_trocr_large_model_v3"

# ===================== CELL 3: Load and Validate Dataset =================

images_dir = os.path.join(DATASET_DIR, "images")
if not os.path.exists(images_dir):
    images_dir = os.path.join(DATASET_DIR, "Images")

labels_path = os.path.join(DATASET_DIR, "labels.csv")
if not os.path.exists(labels_path):
    labels_path = os.path.join(DATASET_DIR, "Labels.csv")

if not os.path.exists(labels_path):
    raise FileNotFoundError(f"labels.csv (or Labels.csv) not found at: {DATASET_DIR}")
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"images/ (or Images/) folder not found at: {DATASET_DIR}")

df = pd.read_csv(labels_path)
print(f"Loaded {len(df)} labeled samples.")
print(df.head())

# Filter out rows where the image file doesn't actually exist
df = df[
    df["Images"].apply(
        lambda x: os.path.exists(os.path.join(images_dir, str(x).strip()))
    )
]
print(f"After filtering missing images: {len(df)} valid samples remain.")

train_df, val_df = train_test_split(
    df, test_size=1 - TRAIN_SPLIT, random_state=42, shuffle=True
)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ===================== CELL 4: Vocabulary Injection =====================
# V3 NEW FEATURE: We inject the top medical drug names directly into the
# tokenizer so each drug = ONE token instead of 4+ subword pieces.
# This dramatically reduces the learning burden on the decoder.

MEDICAL_VOCAB = [
    "Amlodipine",
    "Metformin",
    "Losartan",
    "Atorvastatin",
    "Omeprazole",
    "Pantoprazole",
    "Telmisartan",
    "Rosuvastatin",
    "Metoprolol",
    "Clopidogrel",
    "Aspirin",
    "Enalapril",
    "Lisinopril",
    "Ramipril",
    "Glimepiride",
    "Gliclazide",
    "Voglibose",
    "Teneligliptin",
    "Sitagliptin",
    "Dapagliflozin",
    "Empagliflozin",
    "Pioglitazone",
    "Insulin",
    "Cefuroxime",
    "Amoxicillin",
    "Azithromycin",
    "Clarithromycin",
    "Ciprofloxacin",
    "Levofloxacin",
    "Metronidazole",
    "Fluconazole",
    "Itraconazole",
    "Acyclovir",
    "Hydroxychloroquine",
    "Prednisolone",
    "Dexamethasone",
    "Methylprednisolone",
    "Furosemide",
    "Spironolactone",
    "Hydrochlorothiazide",
    "Chlorthalidone",
    "Bisoprolol",
    "Carvedilol",
    "Nebivolol",
    "Diltiazem",
    "Verapamil",
    "Nifedipine",
    "Digoxin",
    "Warfarin",
    "Dabigatran",
    "Rivaroxaban",
    "Apixaban",
    "Heparin",
    "Enoxaparin",
    "Gabapentin",
    "Pregabalin",
    "Carbamazepine",
    "Phenytoin",
    "Levetiracetam",
    "Valproate",
    "Clonazepam",
    "Alprazolam",
    "Lorazepam",
    "Diazepam",
    "Sertraline",
    "Escitalopram",
    "Fluoxetine",
    "Paroxetine",
    "Amitriptyline",
    "Nortriptyline",
    "Risperidone",
    "Olanzapine",
    "Quetiapine",
    "Haloperidol",
    "Lithium",
    "Tramadol",
    "Codeine",
    "Morphine",
    "Fentanyl",
    "Paracetamol",
    "Ibuprofen",
    "Diclofenac",
    "Naproxen",
    "Celecoxib",
    "Etoricoxib",
    "Rabeprazole",
    "Esomeprazole",
    "Lansoprazole",
    "Ranitidine",
    "Famotidine",
    "Ondansetron",
    "Domperidone",
    "Metoclopramide",
    "Loperamide",
    "Lactulose",
    "Ispaghula",
    "Levothyroxine",
    "Carbimazole",
    "Sildenafil",
    "Tadalafil",
    "Tamsulosin",
    "Finasteride",
    "Dutasteride",
    "Doxazosin",
    "Solifenacin",
    "Oxybutynin",
    "Mirtazapine",
    "Methylcobalamin",
    "Furosemide",
    "Tizanidine",
    "Baclofen",
    "Ceriton",
    "Pizolam",
    "Nexpro",
    "Metmin",
    "Prazocin",
    "Rizarodamol",
    "Injection",
    "Tablet",
    "Capsule",
    "Syrup",
    "Suspension",
    "Solution",
    "OD",
    "BD",
    "TDS",
    "QID",
    "PRN",
    "SOS",
    "STAT",
    "mane",
    "nocte",
    "Lasix",
    "Lasilactone",
    "Aldactone",
    "Epo",
    "Erythropoietin",
    "Ecosprin",
    "Combiflam",
    "Voveran",
    "Cetirizine",
    "Levocetirizine",
    "Fexofenadine",
    "Montelukast",
    "Budesonide",
    "Fluticasone",
    "Salbutamol",
    "Formoterol",
    "Tiotropium",
    "Theophylline",
    "Doxycycline",
    "Roxithromycin",
    "Linezolid",
    "Vancomycin",
    "Meropenem",
    "Ethambutol",
    "Isoniazid",
    "Rifampicin",
    "Pyrazinamide",
    "Artemether",
    "Lumefantrine",
    "Chloroquine",
    "Folic",
    "Ferrous",
    "Cyanocobalamin",
    "Topiramate",
    "Sumatriptan",
    "Rizatriptan",
    "Propranolol",
    "Becosules",
    "Evion",
    "Calpol",
]

print("\n Injecting medical vocabulary into tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-large-handwritten")

# Add new tokens - words not already in the vocabulary
num_added = tokenizer.add_tokens(MEDICAL_VOCAB)
print(f"   Added {num_added} new medical tokens to the vocabulary.")

# ===================== CELL 5: Load TrOCR Model =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}")

feature_extractor = ViTImageProcessor.from_pretrained(
    "microsoft/trocr-large-handwritten"
)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# V3 CRITICAL: Resize embedding table to match new vocabulary size.
# If we injected tokens but don't resize, the model will crash on new words!
if num_added > 0:
    model.decoder.resize_token_embeddings(len(tokenizer))
    print(f"   Decoder embedding table resized to: {len(tokenizer)}")

# Configure special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = len(tokenizer)
model.config.eos_token_id = processor.tokenizer.sep_token_id

# V3 ENCODER UNFREEZING (version-agnostic)
# We use named_modules() to find all ViTLayer instances regardless of nesting.
# This works on any transformers version AND any ViT variant:
#   - ViT-Base: 12 layers → unfreeze last 4 (layers 8-11)
#   - ViT-Large: 24 layers → unfreeze last 4 (layers 20-23)
print("\n Selectively unfreezing top 4 encoder layers...")

# First, freeze everything
for param in model.parameters():
    param.requires_grad = False

# Find ALL ViT transformer layers by type name (works regardless of nesting depth)
all_vit_layers = [
    (name, module)
    for name, module in model.encoder.named_modules()
    if type(module).__name__ == "ViTLayer"
]
print(f"   Found {len(all_vit_layers)} ViT encoder layers total")

# Unfreeze the top 4 layers
UNFREEZE_LAST_N = 4
top_layers = all_vit_layers[-UNFREEZE_LAST_N:]
for name, layer in top_layers:
    for param in layer.parameters():
        param.requires_grad = True
    print(f"   UNFROZEN: encoder.{name}")

# Always unfreeze the decoder (we want it to learn medical vocabulary)
for param in model.decoder.parameters():
    param.requires_grad = True
print("   UNFROZEN: decoder (always)")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(
    f"\n   Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)"
)

model.to(device)
print("\n Model loaded and ready!")

# ===================== CELL 6: Elastic Augmentation Helper ==============
# V3 NEW: Elastic distortion using OpenCV to simulate the way cursive letters
# are stretched, squished, and warped by different doctors' hands.


def elastic_transform(image_pil, alpha=80, sigma=8):
    """
    Applies elastic distortion to a PIL image to simulate cursive variability.
    alpha: intensity of displacement (higher = more warping)
    sigma: smoothness of displacement (higher = smoother, more natural warp)
    """
    img = np.array(image_pil)
    shape = img.shape[:2]  # (H, W)

    # Generate random displacement fields
    dx = np.random.randn(*shape) * alpha
    dy = np.random.randn(*shape) * alpha

    # Smooth the displacement using a Gaussian kernel
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)

    # Create the sampling grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Apply remapping (fill with white background since prescriptions are white)
    distorted = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    return Image.fromarray(distorted)


# ===================== CELL 7: Dataset Class ============================


class PrescriptionCropDataset(Dataset):
    """Custom dataset for prescription crop images with text labels."""

    def __init__(self, dataframe, img_dir, processor, max_length=128, is_train=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        self.is_train = is_train

        # V3: Aggressive augmentation specifically targeting cursive variability
        if is_train:
            self.geometric_aug = T.Compose(
                [
                    # Random perspective simulates different writing angles/slants
                    T.RandomPerspective(distortion_scale=0.15, p=0.5, fill=255),
                    # Small rotation simulates tilted paper
                    T.RandomRotation(degrees=5, expand=True, fill=255),
                ]
            )
            self.photometric_aug = T.Compose(
                [
                    # Color jitter simulates faded ink or photocopies
                    T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.1),
                    # Gaussian blur simulates out-of-focus camera shots
                    T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
                ]
            )
        else:
            self.geometric_aug = None
            self.photometric_aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["Images"]).strip())
        image = Image.open(img_path).convert("RGB")

        if self.is_train:
            # V3: Apply elastic distortion (50% probability)
            if random.random() < 0.5:
                image = elastic_transform(image, alpha=60, sigma=6)
            # Apply geometric augmentations
            image = self.geometric_aug(image)
            # Apply photometric augmentations
            image = self.photometric_aug(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        text = str(row["Text"]).strip()
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        # Replace pad tokens with -100 so they are ignored by the loss function
        labels = [
            lbl if lbl != self.processor.tokenizer.pad_token_id else -100
            for lbl in labels
        ]

        return {"pixel_values": pixel_values, "labels": labels}


train_dataset = PrescriptionCropDataset(
    train_df, images_dir, processor, MAX_TOKEN_LENGTH, is_train=True
)
val_dataset = PrescriptionCropDataset(
    val_df, images_dir, processor, MAX_TOKEN_LENGTH, is_train=False
)

# Quick sanity check
sample = train_dataset[0]
print(f"\n Dataset ready! Sample pixel_values shape: {sample['pixel_values'].shape}")

# ===================== CELL 8: Training =================================

cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# V3: Calculate warmup steps (first 10% of training) for the cosine schedule
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * 2)  # *2 for grad accumulation
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(0.1 * total_steps)
print("\n Training schedule:")
print(f"   Steps/epoch: {steps_per_epoch}")
print(f"   Total steps: {total_steps}")
print(f"   Warmup steps: {warmup_steps}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,  # Effective batch = BATCH_SIZE * 2
    fp16=torch.cuda.is_available(),  # Mixed precision on GPU
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    predict_with_generate=True,
    report_to="none",
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",  # V3: Cosine decay (was linear in v2)
    weight_decay=0.01,
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    dataloader_num_workers=2,
    max_grad_norm=1.0,  # V3: Gradient clipping for stability
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.image_processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # V3: was 3
    compute_metrics=compute_metrics,
)

print("=" * 60)
print("  STARTING FINE-TUNING (V3 - CURSIVE SPECIALIZATION)")
print(f"  Epochs: {EPOCHS} (early stopping patience=5)")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * 2})")
print(f"  Learning rate: {LEARNING_RATE} (cosine with {warmup_steps} warmup steps)")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  New medical tokens: {num_added}")
print("=" * 60)

trainer.train()

# ===================== CELL 9: Save Final Model ========================
print(f"\n Training complete! Saving final model to: {FINAL_MODEL_DIR}")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(" Model saved to Google Drive!")

# ===================== CELL 10: Quick Test ==============================
print("\n--- Quick Test on 5 Validation Samples ---\n")
model.eval()
for i in range(min(5, len(val_df))):
    row = val_df.iloc[i]
    img_path = os.path.join(images_dir, str(row["Images"]).strip())
    image = Image.open(img_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            # V3: Beam search (was greedy in v2) - more accurate predictions
            num_beams=4,
            early_stopping=True,
        )
    predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ground_truth = str(row["Text"]).strip()
    match_emoji = "OK" if predicted.strip().lower() == ground_truth.lower() else "X"
    print(f"[{match_emoji}] GT:   {ground_truth!r}")
    print(f"      Pred: {predicted!r}\n")

print("=" * 60)
print("  ALL DONE!")
print("  Your fine-tuned v3 model is saved at:")
print(f"  {FINAL_MODEL_DIR}")
print()
print("  Next steps:")
print("  1. Download the model folder from Google Drive")
print("  2. Replace the folder at: Medi-Script-AI/models/final_trocr_large_model_v2")
print("     (rename from v3 -> v2, or update the path in segmentation.py to _v3)")
print("=" * 60)
