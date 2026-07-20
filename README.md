# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, which uses a fine-tuned TrOCR model to recognize handwritten medical prescriptions with a focus on open-vocabulary generalization and safety-grounded evaluation.

## Getting Started on Google Colab

Since model training requires a GPU, the recommended environment is **Google Colab**.

### 1. Kaggle API Setup
The datasets are hosted on Kaggle. To download them in Colab, you need your Kaggle API credentials:
1. Go to [Kaggle](https://www.kaggle.com/) and sign in.
2. Click on your profile picture -> **Settings**.
3. Scroll down to the **API** section and click **Create New Token**.
4. This will download a `kaggle.json` file containing your `username` and `key`. Keep this safe!

### 2. Running in Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and set the runtime to **GPU** (`Runtime` -> `Change runtime type` -> select `T4 GPU` or better).
3. Clone this repository in the first cell:
   ```python
   !git clone https://github.com/Mahakisore7/Medi-Script-AI.git
   %cd Medi-Script-AI
   ```
4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
5. Set up Kaggle credentials in Colab (run in a cell and upload your `kaggle.json`):
   ```python
   from google.colab import files
   files.upload() # Upload kaggle.json here
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```
6. Run the data download script:
   ```python
   !python scripts/download_data.py
   ```
7. Start training! (Details to follow as we build the scripts).
