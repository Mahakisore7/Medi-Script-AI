"""
==========================================================================
  Medi-Script-AI: TrOCR Fine-Tuning Script for Google Colab
  -------------------------------------------------------
  This script fine-tunes microsoft/trocr-large-handwritten on YOUR
  custom-labeled prescription crop images.

  Usage: Run each cell in order in Google Colab (GPU runtime).
==========================================================================
"""

# ===================== CELL 1: Install Dependencies =====================
# !pip install -q transformers[torch] datasets evaluate jiwer pillow pandas scikit-learn

# ===================== CELL 2: Mount Google Drive =======================
# from google.colab import drive
# drive.mount('/content/drive')

import os

import evaluate
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

# >>> CHANGE THIS PATH to wherever you uploaded the Roboflow_TrOCR_Dataset folder in Drive <<<
DATASET_DIR = "/content/drive/MyDrive/Medi-Script-AI/Roboflow_TrOCR_Dataset"

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 8  # Colab T4 can handle 8; reduce to 4 if you get OOM errors
LEARNING_RATE = 5e-5
MAX_TOKEN_LENGTH = 64  # Max characters in a label
TRAIN_SPLIT = 0.85  # 85% train, 15% validation

# Output paths (saved to Drive so you don't lose them if Colab disconnects)
OUTPUT_DIR = "/content/drive/MyDrive/Medi-Script-AI/trocr_training_output"
FINAL_MODEL_DIR = "/content/drive/MyDrive/Medi-Script-AI/final_trocr_large_model_v2"

# ===================== CELL 4: Split Dataset ============================

labels_csv = os.path.join(DATASET_DIR, "Labels.csv")
images_dir = os.path.join(DATASET_DIR, "Images")

# Read and clean
df = pd.read_csv(labels_csv)
df = df.dropna(subset=["Text"])
df = df[df["Text"].str.strip() != ""]

# Verify images actually exist
valid_mask = df["Images"].apply(
    lambda x: os.path.exists(os.path.join(images_dir, str(x).strip()))
)
df = df[valid_mask].reset_index(drop=True)

print(f"Total valid samples: {len(df)}")

# Split
train_df, val_df = train_test_split(df, test_size=(1 - TRAIN_SPLIT), random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")

# ===================== CELL 5: Load Model & Processor ===================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not torch.cuda.is_available():
    print("⚠️  WARNING: No GPU detected! Go to Runtime > Change runtime type > GPU (T4)")

print(
    "Loading TrOCR-Large model and processor (this downloads ~1.3GB the first time)..."
)
feature_extractor = ViTImageProcessor.from_pretrained(
    "microsoft/trocr-large-handwritten"
)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-large-handwritten")
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# Configure special tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id

# Freeze the vision encoder to prevent catastrophic forgetting
# (We only want to teach the decoder to output medical words)
print("Freezing the vision encoder (only training the decoder)...")
for param in model.encoder.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

model.to(device)
print("✅ Model loaded and ready!")

# ===================== CELL 6: Dataset Class ============================


class PrescriptionCropDataset(Dataset):
    """Custom dataset for prescription crop images with text labels."""

    def __init__(self, dataframe, img_dir, processor, max_length=64, is_train=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length

        if is_train:
            self.augment = T.Compose(
                [
                    T.RandomRotation(degrees=3, expand=True, fill=255),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
                ]
            )
        else:
            self.augment = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["Images"]).strip())
        image = Image.open(img_path).convert("RGB")

        if self.augment:
            image = self.augment(image)

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
print(f"✅ Dataset ready! Sample pixel_values shape: {sample['pixel_values'].shape}")

# ===================== CELL 7: Training =================================

cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),  # Mixed precision on GPU
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    report_to="none",
    logging_steps=25,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    weight_decay=0.01,
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    dataloader_num_workers=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor.image_processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics,
)

print("=" * 60)
print("  🚀 STARTING FINE-TUNING")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * 2} with grad accum)")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print("=" * 60)

trainer.train()

# ===================== CELL 8: Save Final Model ========================
print(f"\n✅ Training complete! Saving final model to: {FINAL_MODEL_DIR}")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print("✅ Model saved to Google Drive!")

# ===================== CELL 9: Quick Test ===============================
print("\n--- Quick Test on 5 Validation Samples ---\n")
model.eval()
for i in range(min(5, len(val_df))):
    row = val_df.iloc[i]
    img_path = os.path.join(images_dir, str(row["Images"]).strip())
    image = Image.open(img_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ground_truth = str(row["Text"]).strip()
    match_emoji = "✅" if predicted.strip().lower() == ground_truth.lower() else "❌"
    print(f"{match_emoji} GT: {ground_truth!r}")
    print(f"   Pred: {predicted!r}\n")

print("=" * 60)
print("  🎉 ALL DONE!")
print("  Your fine-tuned model is saved at:")
print(f"  {FINAL_MODEL_DIR}")
print()
print("  Next step: Download the model folder from Google Drive")
print("  and place it at: Medi-Script-AI/models/final_trocr_large_model_v2")
print("=" * 60)
