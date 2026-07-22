# Phase 3 — Recognition Engine (Weeks 5–9)

**Owner:** Pod C (Recognition). **Goal:** a fine-tuned TrOCR model that takes a field crop (from Phase 2) and outputs transcribed text. This is the most compute-heavy phase — plan your cloud GPU usage around it.

## 3.1 Scope decision: one shared model, not five

The literature (Biswas et al.) builds a separate pipeline per attribute (name, age, gender, ...). For a semester timeline, fine-tune **one shared TrOCR model** across all field crops, with the field type included as a light signal (e.g., a short prefix token or a separate embedding), and evaluate it broken down by field afterward. One model to train, track, and debug is a large time saving; only split into per-field models later if your shared model clearly struggles on one field type (age and dosage digit sequences are the most likely candidates, given the symbol-confusion issue flagged in your literature review).

## 3.2 Data prep

```python
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def encode(example):
    pixel_values = processor(example["crop_image"], return_tensors="pt").pixel_values
    labels = processor.tokenizer(example["ground_truth_text"], padding="max_length",
                                  max_length=32, truncation=True).input_ids
    labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
    return {"pixel_values": pixel_values.squeeze(), "labels": labels}
```

## 3.3 Fine-tuning script

```python
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate, wandb

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str),
             "wer": wer_metric.compute(predictions=pred_str, references=label_str)}

args = Seq2SeqTrainingArguments(
    output_dir="./trocr-prescription",
    per_device_train_batch_size=4,        # small on purpose — see GPU notes below
    gradient_accumulation_steps=4,        # effective batch size 16
    per_device_eval_batch_size=4,
    fp16=True,                            # mixed precision — critical for 6GB VRAM
    num_train_epochs=15,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                          compute_metrics=compute_metrics)
trainer.train()
```

## 3.4 GPU strategy: local RTX 4050 vs. cloud

Your 4050 has 6GB VRAM — enough for TrOCR-base fine-tuning at small batch size with `fp16` + gradient accumulation (as configured above), which is exactly what you want for **fast local iteration**: debugging the training loop, sanity-checking on a small subset (50–100 images), confirming the metrics pipeline works, before committing to a long run.

For the **full-dataset, full-epoch runs**, use AWS: a `g4dn.xlarge` (T4, 16GB VRAM) lets you raise batch size and cuts training time substantially versus the laptop. Rough workflow:

```bash
# on the EC2 instance
git clone <repo> && cd ml
uv sync
uv run wandb login
uv run python -m recognition.train --config configs/full_run.yaml
```
**Always** set a `sudo shutdown -h +NNN` or a CloudWatch-triggered auto-stop when you launch the instance — this is the single most common way to burn cloud credits by accident. Log everything to W&B so you can `Ctrl+C`/terminate the instance the moment training curves plateau rather than let it run to a fixed epoch count "just in case."

## 3.5 Symbol-awareness (this is a patient-safety issue, not a nice-to-have)

Your literature review specifically flagged `+`, `/`, and digits being confused with letters (`1+1+1` → `1H+1`, `400mg/60mg` → `400mg16mg`). Two things to do now, before post-processing (Phase 4) tries to paper over it:

- Make sure your training data has **good representation** of dosage-style strings with these symbols (check your Phase 1 EDA character-frequency numbers) — if `+` and `/` are rare in your training set, the model will predict them poorly regardless of architecture.
- Log a **per-character-class confusion matrix**, not just overall CER, so you can see specifically whether `+`/`H` and `/`/`1` confusions are actually happening in your model, the same way the literature documented — this tells you whether Phase 4's symbol-correction rules need to be aggressive or light-touch.

## 3.6 Evaluate on the unseen test split — and report both numbers

Report CER/WER on **both** your validation split and your genuinely-unseen test split (Phase 1) side by side, honestly. Expect a gap similar to the literature (validation CER meaningfully better than unseen-test CER) — that gap *is* your generalization result and is worth discussing in your final report rather than hiding.

## Definition of done for Phase 3

- [ ] TrOCR fine-tuned, checkpoints versioned in DVC
- [ ] CER/WER logged in W&B for every run, validation and unseen-test reported separately
- [ ] Per-field breakdown of CER (not just an overall number)
- [ ] Confusion analysis done specifically for `+`, `/`, digit characters
- [ ] Best checkpoint wrapped in a simple `Recognizer.transcribe(crop) -> str` interface for Phase 5
