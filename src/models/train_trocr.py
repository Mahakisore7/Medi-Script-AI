import os
import torch
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import evaluate
from src.data.dataset import HandwritingDataset

# Metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred, processor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

def train_model(train_df, eval_df, output_dir="./models/trocr-finetuned", epochs=3, batch_size=4):
    """
    Trains the TrOCR model.
    """
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    train_dataset = HandwritingDataset(train_df, processor)
    eval_dataset = HandwritingDataset(eval_df, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        num_train_epochs=epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False, # We want lower CER
        report_to="none", # Disable wandb to prevent interactive prompt hanging
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor, # Pass feature extractor as tokenizer
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    # By passing resume_from_checkpoint=True, the Trainer will automatically 
    # find the latest checkpoint in output_dir (e.g. checkpoint-3500) and resume from it!
    # If no checkpoint exists, it safely starts from scratch.
    trainer.train(resume_from_checkpoint=True)
    
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    return model, processor
