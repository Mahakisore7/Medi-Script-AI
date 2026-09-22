import os

import evaluate
import pandas as pd
import torchvision.transforms as T
from PIL import Image
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


class RxHandBDDataset(Dataset):
    """
    Dataset class specifically for RxHandBD which uses 'Images' and 'Text' columns
    """

    def __init__(self, manifest_csv: str, img_dir: str, processor, is_train: bool = False):
        self.df = pd.read_csv(manifest_csv)
        self.processor = processor
        self.img_dir = img_dir
        self.is_train = is_train

        # Ensure we don't have NaN texts
        self.df = self.df.dropna(subset=["Text"])

        if self.is_train:
            self.transform = T.Compose(
                [
                    T.RandomRotation(degrees=3, expand=True, fill=255),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["Images"]).strip()
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            str(row["Text"]).strip(), padding="max_length", max_length=32, truncation=True
        ).input_ids

        labels = [lbl if lbl != self.processor.tokenizer.pad_token_id else -100 for lbl in labels]

        return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(pred, processor):
    cer_metric = evaluate.load("cer")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def train():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to RxHandBD extracted folder"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print("Loading TrOCR-Large processor and model...")
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-large-handwritten")
    processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Freeze the vision encoder to prevent catastrophic forgetting
    print("Freezing the vision encoder...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Prepare datasets
    train_csv = os.path.join(args.data_dir, "Train_Label.csv")
    train_img_dir = os.path.join(args.data_dir, "Train_Set")
    val_csv = os.path.join(args.data_dir, "Test_Labels.csv")
    val_img_dir = os.path.join(args.data_dir, "Test_Set")

    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found!")
        return

    print("Building datasets...")
    train_ds = RxHandBDDataset(train_csv, train_img_dir, processor, is_train=True)
    val_ds = RxHandBDDataset(val_csv, val_img_dir, processor, is_train=False)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir="models/rxhandbd_trocr_large",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,  # Use mixed precision if GPU allows
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        report_to="none",
        logging_steps=50,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor.image_processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    print("Starting fine-tuning! This may take several hours depending on your GPU...")
    trainer.train()

    print("Training complete! Saving final model...")
    trainer.save_model("models/final_trocr_large_model")
    processor.save_pretrained("models/final_trocr_large_model")
    print("Done! The new fine-tuned model is ready at models/final_trocr_large_model")


if __name__ == "__main__":
    train()
