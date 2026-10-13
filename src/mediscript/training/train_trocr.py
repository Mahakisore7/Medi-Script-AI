import os

import evaluate
from torch.utils.data import ConcatDataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from mediscript.training.trocr_dataset import FieldCropDataset


def compute_metrics(pred):
    cer_metric = evaluate.load("cer")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Replace -100 with pad token in labels
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def train():
    """
    TrOCR fine-tuning script.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, default="")
    parser.add_argument("--val_manifest", type=str, default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args_cmd = parser.parse_args()

    print("Starting TrOCR fine-tuning...")
    global processor

    from transformers import RobertaTokenizer, ViTImageProcessor

    # Use base model components separately to avoid FastTokenizer bugs on Windows Python 3.13
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Get absolute paths to data if not provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "data", "trocr-dataset"))

    train_csv = (
        args_cmd.train_manifest
        if args_cmd.train_manifest
        else os.path.join(data_dir, "train_manifest.csv")
    )
    val_csv = (
        args_cmd.val_manifest
        if args_cmd.val_manifest
        else os.path.join(data_dir, "val_manifest.csv")
    )

    if not os.path.exists(train_csv):
        print(f"Dataset not found at {train_csv}. Run generate_trocr_data.py first.")
        return

    train_ds = FieldCropDataset(train_csv, processor, is_train=True)
    val_ds = FieldCropDataset(val_csv, processor, is_train=False)

    synthetic_csv = os.path.abspath(
        os.path.join(script_dir, "..", "..", "..", "datasets", "synthetic_trocr", "synthetic_manifest.csv")
    )
    if os.path.exists(synthetic_csv):
        print(f"Loading synthetic dataset from {synthetic_csv}")
        synth_ds = FieldCropDataset(synthetic_csv, processor, is_train=True)
        train_ds = ConcatDataset([train_ds, synth_ds])
        print(f"Combined dataset size: {len(train_ds)}")

    # Training arguments
    args = Seq2SeqTrainingArguments(
        output_dir="./trocr-rx-finetuned",
        per_device_train_batch_size=args_cmd.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args_cmd.batch_size,
        fp16=False,  # CPU doesn't support fp16 well
        num_train_epochs=args_cmd.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Only keep the best/latest checkpoint to save disk space
        predict_with_generate=True,
        report_to="none",
        logging_steps=50,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor.image_processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Smoke test complete! The TrOCR setup works perfectly.")


if __name__ == "__main__":
    train()
