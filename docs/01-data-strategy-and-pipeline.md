# Phase 1 — Data Strategy & Pipeline (Weeks 2–4)

**Owner:** Pod A (Data & MLOps). **Goal:** a versioned, annotated, augmented dataset that the segmentation and recognition pods can both build on, sitting in DVC/S3, not on anyone's laptop.

## 1.1 Field taxonomy — agree on this before collecting anything

Define exactly what a "field" is, since the segmentation model (Phase 2) and recognition model (Phase 3) both depend on this being fixed:

```yaml
fields:
  - patient_name
  - age
  - gender
  - symptoms
  - medicine_and_dosage   # kept together — dosage notation (1+1+1, 400mg/60mg) is easiest
                          # to keep next to the medicine name during recognition, split later
                          # in post-processing if needed
```

Keep it to 5 field types for the MVP. Splitting `medicine` and `dosage` into two separate boxes is future scope if time allows (see `08-stretch-goals.md`).

## 1.2 Collection (per the strategy signed off in Phase 0)

- **Synthetic-but-real-handwriting collection:** build a simple Google Form / printed template with 5 labeled blank fields, ask each volunteer to fill in *fabricated* patient details across, say, 15–20 variations each. With 4+ team members each recruiting 10–15 volunteers, you can realistically reach several hundred to ~1,000 labeled prescriptions — comparable in scale to Ali et al.'s ~1,000 real-prescription starting point in the literature you reviewed, before augmentation.
- Scan/photograph consistently: good lighting, flat surface, same phone camera app if possible, so your images aren't fighting camera-quality variance on top of handwriting variance.
- **Public supplement:** download IAM-OnDB and any Roboflow "handwritten prescription" datasets for pretraining/augmentation diversity — these won't match your exact field taxonomy but are useful for the recognition model to see more handwriting styles than your volunteer pool alone provides.

## 1.3 Annotation

Use **Label Studio** (free, self-hostable, or their free cloud tier) to draw bounding boxes around each of the 5 field types per image, and transcribe the ground-truth text for each box.

```bash
uv add label-studio
uv run label-studio start
```
Export format: choose YOLO format directly if available (bounding boxes) plus a separate transcription CSV (`image_id, field_type, bbox, ground_truth_text`) — you need both the box (for Phase 2) and the transcription (for Phase 3) per crop.

**Annotation discipline that pays off later:** have 2 people label the same 20 images independently early on and compare — if your team disagrees on where a box boundary goes, your model will struggle to learn a consistent signal. Write a one-page annotation guide once you've resolved disagreements.

## 1.4 The three-way split — do this now, not at evaluation time

Following the evaluation discipline that the literature review flagged as the single most honest practice (Ali et al.'s three-way split):

1. **Train** (~70%)
2. **Validation** (~15%) — same handwriting styles/vocabulary as train, used for tuning
3. **Genuinely unseen test** (~15%) — held out from *specific volunteers* and *specific medicine names* your model never sees in training, not just a random shuffle. This is what makes your reported accuracy honest instead of optimistic.

Do the split at the *person* and *vocabulary* level, not the image level — if the same volunteer's handwriting appears in both train and test, your test accuracy will look better than real-world performance will actually be.

## 1.5 Augmentation

Use `albumentations` for pixel-level augmentation, matching what worked in the literature (rotation, contrast, blur, noise, elastic transform):

```python
import albumentations as A

train_transform = A.Compose([
    A.Rotate(limit=5, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(1, 3), p=0.2),
    A.GaussNoise(var_limit=(5, 30), p=0.3),
    A.ElasticTransform(alpha=1, sigma=15, p=0.2),
])
```
Augment the **training** split only — never augment validation/test, or you'll contaminate your evaluation.

## 1.6 Version with DVC

```bash
cd data/
dvc add raw/prescriptions
git add raw/prescriptions.dvc .gitignore
git commit -m "data: add raw prescription collection v1"
dvc push
```
Anyone on the team can now `dvc pull` to fetch exactly the data version referenced by their current git branch — no more "whose USB stick has the latest dataset" problems.

## 1.7 EDA notebook

In `ml/notebooks/`, do a short exploratory pass: field-length distributions, character frequency (are `+`, `/`, digits well represented?), image quality spread, class balance across the 5 field types, per-volunteer sample counts. This isn't busywork — it directly informs whether your augmentation and lexicon (Phase 4) need adjusting before you sink weeks into training.

## Definition of done for Phase 1

- [ ] Field taxonomy frozen and documented
- [ ] At least a few hundred labeled prescriptions collected across the team
- [ ] Annotation guide written, Label Studio exports validated by a second annotator on a sample
- [ ] Train/val/**unseen-test** split done at person+vocabulary level, not just randomly
- [ ] Augmentation pipeline implemented and applied to train split only
- [ ] Dataset versioned in DVC, pushed to S3, pullable by any teammate
- [ ] EDA notebook committed with findings that Pod B/C/D can act on
