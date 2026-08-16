import os
import time
import glob
import csv
import cv2
import base64
import requests
from pathlib import Path
from ultralytics import YOLO

# =========================================================================
# CONFIGURATION
# =========================================================================
ROBOFLOW_DATASET_DIR = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\Medi-Script-AI\Medicine_Only_Workspace\RoboflowDataset"
YOLO_MODEL_PATH = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\Medi-Script-AI\runs\detect\yolo_real_world_best.pt"

OUTPUT_DATASET_DIR = Path("Roboflow_TrOCR_Dataset")
OUTPUT_IMAGES_DIR = OUTPUT_DATASET_DIR / "Images"
OUTPUT_CSV_PATH = OUTPUT_DATASET_DIR / "Labels.csv"

OLLAMA_MODEL = "llava"

# =========================================================================
# INITIALIZATION
# =========================================================================
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize YOLO
print(f"[INFO] Loading YOLO model from: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================================================================
# SCRIPT
# =========================================================================
def ask_ollama_for_transcription(image_path: str) -> str:
    """Sends the cropped image to local Ollama Vision model."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        url = "http://localhost:11434/api/generate"
        data = {
            "model": OLLAMA_MODEL,
            "prompt": "You are an expert medical transcriptionist. Read the text in this cropped image of a prescription. Output EXACTLY what is written, including any numbers, symbols, abbreviations (like Tab, Syp, Cap), and dosages (like 1-0-1, TDS, SOS). Do NOT add any conversational text. Do NOT add markdown formatting. Just output the plain text string that you see.",
            "stream": False,
            "images": [encoded_image]
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"[!] Ollama API Error: {response.text}")
            return ""
            
    except requests.exceptions.ConnectionError:
        print("[!] Connection Error: Is Ollama running? Start Ollama desktop app first!")
        return ""
    except Exception as e:
        print(f"[!] Local LLM Error on {image_path}: {e}")
        return ""

def main():
    # Gather all jpg images recursively
    image_paths = glob.glob(os.path.join(ROBOFLOW_DATASET_DIR, "**", "*.jpg"), recursive=True)
    print(f"\nFound {len(image_paths)} images in the Roboflow dataset.")

    # Prepare CSV file
    file_exists = OUTPUT_CSV_PATH.exists()
    with open(OUTPUT_CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["Images", "Text"]) # Header required by TrOCR training script

        crop_counter = 0

        for img_path in image_paths:
            print(f"\nProcessing {os.path.basename(img_path)}...")
            
            try:
                # 1. Run YOLO to find medicine boxes
                results = yolo_model(img_path, verbose=False)
                original_img = cv2.imread(img_path)
                
                if original_img is None:
                    continue

                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        
                        # Crop the image
                        crop = original_img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                            
                        # Save the crop temporarily
                        crop_filename = f"crop_{int(time.time())}_{crop_counter}.jpg"
                        crop_filepath = str(OUTPUT_IMAGES_DIR / crop_filename)
                        cv2.imwrite(crop_filepath, crop)
                        
                        # 2. Run Local Ollama
                        print(f"  -> Sending crop {crop_counter} to local Llama-3.2-Vision...")
                        text = ask_ollama_for_transcription(crop_filepath)
                        
                        if text:
                            # 3. Save to CSV
                            writer.writerow([crop_filename, text])
                            csv_file.flush() 
                            print(f"     Got: '{text}'")
                        else:
                            print(f"     Failed to transcribe crop {crop_counter}. Skipping.")
                            os.remove(crop_filepath)
                            
                        crop_counter += 1
                        
            except Exception as e:
                print(f"[!] Error processing {img_path}: {e}")

    print("\n=======================================================")
    print(f"Done! Extracted {crop_counter} crops.")
    print(f"Dataset is ready at: {OUTPUT_DATASET_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    main()
