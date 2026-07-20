import os
import sys
# Add the project root to the Python path so 'src' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.models.train_trocr import train_model

def parse_iam_words(words_txt_path, images_dir):
    """
    Parses the IAM words.txt file and returns a DataFrame suitable for training.
    """
    data = []
    # Read the words.txt file
    with open(words_txt_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 9:
                word_id = parts[0]
                status = parts[1] # 'ok' or 'err'
                
                # Only use words where the segmentation was 'ok'
                if status == 'ok':
                    text = " ".join(parts[8:])
                    
                    # Construct image path
                    # Format: a01-000u-00-00
                    # Directory structure: a01/a01-000u/a01-000u-00-00.png
                    split_id = word_id.split("-")
                    dir1 = split_id[0]
                    dir2 = f"{split_id[0]}-{split_id[1]}"
                    img_path = os.path.join(images_dir, dir1, dir2, f"{word_id}.png")
                    
                    if os.path.exists(img_path):
                        data.append({"file_name": img_path, "text": text})
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "iam_dataset")
    
    words_txt = os.path.join(DATA_DIR, "iam_words", "words.txt")
    images_dir = os.path.join(DATA_DIR, "iam_words", "words")
    
    if not os.path.exists(words_txt):
        print(f"Error: {words_txt} not found. Please run download_data.py first.")
        exit(1)
        
    print("Parsing IAM dataset...")
    df = parse_iam_words(words_txt, images_dir)
    print(f"Loaded {len(df)} images.")
    
    # Split train/eval (90% / 10%)
    train_df = df.sample(frac=0.9, random_state=42)
    eval_df = df.drop(train_df.index)
    
    # Reset index
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    
    print(f"Training on {len(train_df)} samples, evaluating on {len(eval_df)} samples.")
    
    # Run training
    train_model(train_df, eval_df, output_dir="./models/trocr-iam-pretrained", epochs=3, batch_size=8)
