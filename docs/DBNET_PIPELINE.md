# DBNet + TrOCR Pipeline Branch

This branch introduces a text-detection-first alternative to the original field-YOLO pipeline.

## Goal

Instead of asking a detector to directly predict semantic fields such as `name` or `medicine`, this branch uses:

1. `DBNet` to detect text regions
2. `TrOCR` to recognize each detected region
3. a heuristic field mapper to assign OCR results into:
   - `name`
   - `age`
   - `gender`
   - `symptoms`
   - `medicine`

## Why This Branch Exists

The original YOLO detector was trained on synthetic field annotations. It works as a baseline, but it struggles on real prescriptions with variable layouts.

DBNet is a better fit for the intermediate goal of:

- detecting all text lines or regions
- handling more flexible document layouts
- separating text detection from field assignment

## How To Run

Place a DBNet ONNX model at:

`models/dbnet.onnx`

Or provide an explicit path through an environment variable.

Run:

```powershell
$env:SEGMENTATION_MODE="dbnet"
$env:DBNET_MODEL_PATH="C:\path\to\dbnet.onnx"
uv run --project ml python demo.py
```

If `DBNET_MODEL_PATH` is not set, the code will look for:

`models/dbnet.onnx`

## Current Implementation Notes

- The DBNet detector uses OpenCV's `dnn_TextDetectionModel_DB`.
- Detected text regions are sorted top-to-bottom.
- Each crop is passed through TrOCR.
- A heuristic mapper assigns OCR outputs into the structured prescription fields.

## Current Limitation

This branch adds the architecture and integration, but accuracy still depends on:

- the quality of the DBNet ONNX model you provide
- the quality of TrOCR on prescription handwriting
- the current heuristic field mapper, which should later be replaced by a stronger learned or rule-enhanced layout model
