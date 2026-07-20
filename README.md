# SafeRx-OCR: Handwriting Recognition for Medical Prescriptions

This repository contains the codebase for the SafeRx-OCR thesis project, which uses a fine-tuned TrOCR model to recognize handwritten medical prescriptions with a focus on open-vocabulary generalization and safety-grounded evaluation.

## Getting Started on Google Colab

Since model training requires a GPU, the recommended environment is **Google Colab**.

### 1. Authenticating GitHub (For Private Repositories)
Since this repository is private, standard `git clone` commands will fail in Colab. You must authenticate using a Personal Access Token (Classic).

1. Go to GitHub -> Settings -> Developer Settings -> Personal access tokens -> **Tokens (classic)**.
2. Click **Generate new token (classic)**.
3. Check the **`repo`** checkbox (crucial step, or you will get a 403 error).
4. Generate the token and copy it (starts with `ghp_...`).

### 2. Kaggle API Setup
The datasets are hosted on Kaggle. To download them in Colab, you need your Kaggle API credentials:
1. Go to [Kaggle](https://www.kaggle.com/) and sign in.
2. Click on your profile picture -> **Settings**.
3. Scroll down to the **API** section and click **Create New Token**.
4. You will get a `username` and an API `key` (a long string of characters). 

### 3. Running in Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and set the runtime to **GPU** (`Runtime` -> `Change runtime type` -> select `T4 GPU` or better).
3. Clone this repository in the first cell, replacing `<USERNAME>` with your GitHub username and `<TOKEN>` with the classic token you generated in Step 1:
   ```python
   !git clone -b feature/ocr-setup https://<USERNAME>:<TOKEN>@github.com/Mahakisore7/Medi-Script-AI.git
   %cd Medi-Script-AI
   ```
4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
5. Set up Kaggle credentials. Run this code in a new cell, replacing `YOUR_USERNAME` and `YOUR_TOKEN` with your actual Kaggle details:
   ```python
   import os
   import json

   os.makedirs('/root/.kaggle', exist_ok=True)
   
   kaggle_creds = {
       "username": "YOUR_USERNAME",
       "key": "YOUR_TOKEN"
   }
   
   with open('/root/.kaggle/kaggle.json', 'w') as f:
       json.dump(kaggle_creds, f)
       
   os.chmod('/root/.kaggle/kaggle.json', 0o600)
   print("Kaggle credentials set up successfully!")
   ```
6. Run the data download script:
   ```python
   !python scripts/download_data.py
   ```
7. Start training!
   ```python
   !python scripts/run_iam_pretraining.py
   ```
