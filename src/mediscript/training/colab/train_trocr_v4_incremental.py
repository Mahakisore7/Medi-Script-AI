"""
==========================================================================
  Medi-Script-AI: TrOCR Incremental Fine-Tuning
  -----------------------------------------------------------------------
  Use this script to train your ALREADY TRAINED v3 model on new synthetic
  dosage data, without starting from scratch.

  Usage: Run each cell in order in Google Colab (GPU runtime T4 or better).
==========================================================================
"""

# ===================== CELL 1: Install Dependencies =====================
# !pip install -q transformers[torch] datasets evaluate jiwer pillow pandas scikit-learn opencv-python-headless

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

# ===================== CONFIGURATION =====================
# This is where your synthetic images and labels are stored
DATASET_DIR = "/content/drive/MyDrive/Medi-Script-AI/Roboflow_TrOCR_Dataset"

# This is where your uploaded v3 model should be
BASE_MODEL_DIR = "/content/drive/MyDrive/Medi-Script-AI/final_trocr_large_model_v3"

# Output for the incrementally trained model
OUTPUT_DIR = "/content/drive/MyDrive/Medi-Script-AI/trocr_incremental_training_output"
FINAL_MODEL_DIR = (
    "/content/drive/MyDrive/Medi-Script-AI/final_trocr_large_model_v4_dosage"
)

EPOCHS = 5  # Only need a few epochs since it's already trained!
BATCH_SIZE = 4
LEARNING_RATE = 1e-6  # VERY low learning rate to prevent catastrophic forgetting
MAX_TOKEN_LENGTH = 128
TRAIN_SPLIT = 0.85

# ===================== CELL 3: Load Dataset =================
images_dir = os.path.join(DATASET_DIR, "images")
labels_path = os.path.join(DATASET_DIR, "labels.csv")

if not os.path.exists(BASE_MODEL_DIR):
    raise FileNotFoundError(
        f"CRITICAL ERROR: Could not find your base v3 model at {BASE_MODEL_DIR}. Did you upload it?"
    )

df = pd.read_csv(labels_path)
print(f"Loaded {len(df)} labeled samples.")

# Filter missing
df = df[
    df["Images"].apply(
        lambda x: os.path.exists(os.path.join(images_dir, str(x).strip()))
    )
]
print(f"Valid samples: {len(df)}")

train_df, val_df = train_test_split(
    df, test_size=1 - TRAIN_SPLIT, random_state=42, shuffle=True
)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ===================== CELL 4: Load v3 Model & Processor =====================
print("\n Loading existing v3 model from Drive...")
# We load directly from the v3 folder! No vocabulary injection needed because it's already in there.
tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL_DIR)
feature_extractor = ViTImageProcessor.from_pretrained(BASE_MODEL_DIR)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_DIR)

# Configure special tokens (just to be safe)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = len(tokenizer)
model.config.eos_token_id = processor.tokenizer.sep_token_id

# FREEZE STRATEGY:
# Freeze the ENTIRE encoder (it already knows how to see cursive).
# Unfreeze only the decoder (so it learns the syntax of 1-0-1 x 5days).
print("\n Freezing encoder, unfreezing decoder...")
for param in model.encoder.parameters():
    param.requires_grad = False

for param in model.decoder.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f" Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("\n Model ready for incremental training!")


# ===================== CELL 5: Dataset Class ============================
class PrescriptionCropDataset(Dataset):
    def __init__(self, dataframe, img_dir, processor, max_length=128, is_train=False):
        self.df = dataframe
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        self.is_train = is_train

        if is_train:
            self.geometric_aug = T.Compose(
                [
                    T.RandomPerspective(distortion_scale=0.15, p=0.5, fill=255),
                    T.RandomRotation(degrees=5, expand=True, fill=255),
                ]
            )
            self.photometric_aug = T.Compose(
                [
                    T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.1),
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
            image = self.geometric_aug(image)
            image = self.photometric_aug(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        text = str(row["Text"]).strip()
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_length, truncation=True
        ).input_ids

        labels = [
            lbl if lbl != self.processor.tokenizer.pad_token_id else -100
            for lbl in labels
        ]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


train_dataset = PrescriptionCropDataset(
    train_df, images_dir, processor, max_length=MAX_TOKEN_LENGTH, is_train=True
)
val_dataset = PrescriptionCropDataset(
    val_df, images_dir, processor, max_length=MAX_TOKEN_LENGTH, is_train=False
)

# ===================== CELL 6: Metrics & Training =======================
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
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=True,
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    processing_class=processor.image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n Starting Incremental Training...")
trainer.train()

print(f"\n Saving final trained model to {FINAL_MODEL_DIR}...")
processor.save_pretrained(FINAL_MODEL_DIR)
trainer.save_model(FINAL_MODEL_DIR)
print("\n Done! Your model is now updated with dosage syntax.")
