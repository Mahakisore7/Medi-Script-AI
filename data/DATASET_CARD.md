# Dataset Card for rx-recognition

## Dataset Description

- **Repository:** `[PENDING KAGGLE SLUG]`
- **Paper:** (Your Thesis Name Here)
- **Point of Contact:** `rx-recognition` Project Team

### Dataset Summary
This dataset contains bounding box and transcription annotations for handwritten medical prescriptions. It is a unified corpus built from public datasets (such as the Roboflow Doctors' Prescriptions Handwriting dataset) and processed into a single schema for the `rx-recognition` pipeline. No protected health information (PHI) or real patient records are included.

## Dataset Structure

### Data Instances
A typical data point consists of an image of a handwritten prescription and an associated JSON annotation defining bounding boxes and transcriptions for 5 specific fields:
1. Name
2. Age
3. Gender
4. Symptoms
5. Medicine + Dosage

### Data Splits
To ensure true generalization capabilities (avoiding the pitfalls identified in Ali et al.), the data is split in a strict 3-way manner:
- **Train:** Bulk of the data.
- **Validation:** Randomly sampled from the same distribution as the training set.
- **Unseen Test:** Held out entirely by writer ID or vocabulary (words not seen in training), providing a true metric for cross-institution generalization.

## Dataset Creation

### Curation Rationale
Medical prescription recognition requires in-domain data. To adhere to strict ethical guidelines for a credit-course project, no raw clinical data was collected. Data was curated entirely from pre-existing public datasets and synthetic generators.

### Source Data
- **Roboflow Universe:** Doctors' Prescriptions Handwriting (~580 images)
- **IAM-OnDB:** (For base handwriting features)

## Considerations for Using the Data

### Social Impact of Dataset
The dataset explicitly excludes actual patient data to eliminate the risk of PHI leaks, prioritizing ethical ML research principles.

### Discussion of Biases
The dataset is primarily restricted to block-print or mildly cursive English handwriting. It may underrepresent certain regional handwriting variations and does not yet capture complex multilingual scripts (e.g., Devanagari, Bengali, Urdu) outside of the stretch goals.
