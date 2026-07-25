import os

import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from rx_recognition.trocr_dataset import FieldCropDataset


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
    Local smoke test for TrOCR fine-tuning.
    We run only 2 steps to ensure everything compiles and runs before taking this to Kaggle.
    """
    print("Starting TrOCR smoke test...")
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

    # Get absolute paths to data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "trocr-dataset"))

    train_csv = os.path.join(data_dir, "train_manifest.csv")
    val_csv = os.path.join(data_dir, "val_manifest.csv")

    if not os.path.exists(train_csv):
        print(f"Dataset not found at {train_csv}. Run generate_trocr_data.py first.")
        return

    train_ds = FieldCropDataset(train_csv, processor)
    val_ds = FieldCropDataset(val_csv, processor)

    # Smoke test arguments
    args = Seq2SeqTrainingArguments(
        output_dir="./trocr-rx-finetuned",
        per_device_train_batch_size=1,  # small — raise cautiously
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        fp16=False,  # disable fp16 since we don't have GPU
        max_steps=2,  # ONLY 2 STEPS FOR SMOKE TEST!
        eval_strategy="no",
        save_strategy="no",
        predict_with_generate=True,
        report_to="none",  # Disable wandb for local smoke test
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor.image_processor,
    )

    trainer.train()
    print("Smoke test complete! The TrOCR setup works perfectly.")


if __name__ == "__main__":
    train()
