# Phase 2 — Preprocessing & Field Localization (Weeks 3–6)

**Owner:** Pod B (Vision). **Goal:** given a raw prescription photo, output cropped, cleaned images for each of the 5 field types, ready to feed to the recognition model in Phase 3.

## 2.1 Preprocessing (OpenCV)

```python
import cv2
import numpy as np

def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # deskew
    coords = np.column_stack(np.where(denoised > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = denoised.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed
```
Keep binarization **optional and configurable** — TrOCR (Phase 3) is trained on natural handwriting images, not pure black/white binarized ones, so hard binarization can sometimes hurt the recognition stage even though it helps some classical OCR baselines (this is a real, non-obvious interaction between your two modules — test both ways once Phase 3 exists rather than assuming binarization is always good).

## 2.2 Why a detector instead of Mask R-CNN

The papers you reviewed (Ali et al., Biswas et al.) use Mask R-CNN / CNN-SVM for segmentation. For your MVP, swap this for **YOLOv8/YOLO11 object detection**: you need a bounding box per field, not a pixel-accurate mask, and Detectron2 (the usual Mask R-CNN implementation) is a genuine pain to install on Windows without WSL. This is a defensible substitution to state explicitly in your report — same problem, more practical tool for your timeline and platform.

## 2.3 Train the field detector

```bash
cd ml/segmentation
uv add ultralytics
```
Convert your Label Studio export to YOLO format (image + `.txt` per image with `class x_center y_center width height`, normalized 0–1). Ultralytics has a built-in Label Studio → YOLO converter, or write a ~30-line script if your export format differs.

`data.yaml`:
```yaml
path: ../../data/segmentation
train: images/train
val: images/val
names:
  0: patient_name
  1: age
  2: gender
  3: symptoms
  4: medicine_and_dosage
```

Train:
```bash
uv run yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 device=0
```
Start with `yolov8n` (nano) — it's small enough to train fast on your RTX 4050 locally with your few-hundred-image dataset, and field detection on a fairly structured layout (prescriptions) doesn't need a huge model. Only move to `yolov8s`/`yolov8m` if nano's mAP is clearly insufficient.

## 2.4 Evaluate

```bash
uv run yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml
```
Track mAP@0.5 per field class in W&B (`uv add wandb` integration is built into Ultralytics — pass `project=` and `name=` to the train command). Watch for classes with very few training examples (check your Phase 1 EDA) — `age` and `gender` fields are usually small and easy to miss; if mAP is poor for a specific field, that's almost always a data problem, not a model problem, at this scale.

## 2.5 Wrap it as a clean module

Pod C (Recognition) shouldn't need to know anything about YOLO internals — give them a simple interface:

```python
# ml/segmentation/detector.py
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class FieldCrop:
    field_type: str
    image: "np.ndarray"
    confidence: float

class FieldDetector:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def detect(self, image) -> list[FieldCrop]:
        results = self.model(image)[0]
        crops = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crops.append(FieldCrop(cls_name, image[y1:y2, x1:x2], float(box.conf)))
        return crops
```
This is the integration contract for Phase 5 — write it now even before it's perfect, so Pod C can start building against it.

## Definition of done for Phase 2

- [ ] Preprocessing function committed, tested on a sample of messy real photos (shadows, skew, low light)
- [ ] YOLO field detector trained, mAP@0.5 logged per field in W&B
- [ ] Weights versioned in DVC (not committed to git — model weights are binary and large)
- [ ] `FieldDetector` module with a stable `detect(image) -> list[FieldCrop]` interface, so Pod C can integrate against it without needing to understand YOLO
