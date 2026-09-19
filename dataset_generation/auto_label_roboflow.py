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
API_KEY = "AQ.Ab8RN6Ka_V0eigMjdIlYOdNPaLN4w9w2WonAEuMr_SBzTGw_4g"
ROBOFLOW_DATASET_DIR = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\Medi-Script-AI\Medicine_Only_Workspace\RoboflowDataset"
YOLO_MODEL_PATH = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\Medi-Script-AI\runs\detect\yolo_real_world_best.pt"

OUTPUT_DATASET_DIR = Path("Roboflow_TrOCR_Dataset")
OUTPUT_IMAGES_DIR = OUTPUT_DATASET_DIR / "Images"
OUTPUT_CSV_PATH = OUTPUT_DATASET_DIR / "Labels.csv"

# Free tier limits: 15 requests per minute -> 1 request every 4 seconds.
SLEEP_TIME_SECONDS = 4.1

# =========================================================================
# INITIALIZATION
# =========================================================================
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize YOLO
print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)


# =========================================================================
# SCRIPT
# =========================================================================
def ask_gemini_for_transcription(image_path: str) -> str:
    """Sends the cropped image to Gemini with auto-retries for 503 High Demand."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "You are an expert medical transcriptionist. Read the text in this cropped image of a prescription. Output EXACTLY what is written, including any numbers, symbols, abbreviations (like Tab, Syp, Cap), and dosages (like 1-0-1, TDS, SOS). Do NOT add any conversational text. Do NOT add markdown formatting. Just output the plain text string that you see."
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image,
                            }
                        },
                    ]
                }
            ]
        }

        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(url, headers=headers, json=data)
            response_json = response.json()

            if "candidates" in response_json:
                return response_json["candidates"][0]["content"]["parts"][0][
                    "text"
                ].strip()
            elif "error" in response_json and response_json["error"].get("code") == 503:
                print(
                    f"     [!] High Demand (503). Retrying in 15 seconds... (Attempt {attempt+1}/{max_retries})"
                )
                time.sleep(15)
                continue
            else:
                print(f"[!] Unexpected Gemini API Response: {response_json}")
                return ""

        print(f"[!] Failed after {max_retries} retries due to server load.")
        return ""
    except Exception as e:
        print(f"[!] Gemini API Error on {image_path}: {e}")
        return ""


def main():
    # Gather all jpg images recursively
    image_paths = glob.glob(
        os.path.join(ROBOFLOW_DATASET_DIR, "**", "*.jpg"), recursive=True
    )
    print(f"Found {len(image_paths)} images in the Roboflow dataset.")

    # Prepare CSV file
    file_exists = OUTPUT_CSV_PATH.exists()

    start_crop_index = 0
    if file_exists:
        with open(OUTPUT_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0].startswith("crop_"):
                    try:
                        idx = int(row[0].split("_")[-1].split(".")[0])
                        start_crop_index = max(start_crop_index, idx + 1)
                    except:
                        pass
        if start_crop_index > 0:
            print(
                f"\n[!] Found existing dataset. Auto-resuming from crop {start_crop_index}..."
            )

    with open(OUTPUT_CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                ["Images", "Text"]
            )  # Header required by TrOCR training script

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

                        if crop_counter < start_crop_index:
                            crop_counter += 1
                            continue

                        # Save the crop temporarily
                        crop_filename = f"crop_{int(time.time())}_{crop_counter}.jpg"
                        crop_filepath = str(OUTPUT_IMAGES_DIR / crop_filename)
                        cv2.imwrite(crop_filepath, crop)

                        # 2. Ask Gemini for the text
                        print(f"  -> Sending crop {crop_counter} to Gemini...")
                        text = ask_gemini_for_transcription(crop_filepath)

                        if text:
                            # 3. Save to CSV
                            writer.writerow([crop_filename, text])
                            # Flush immediately so we don't lose data if we stop the script early
                            csv_file.flush()
                            print(f"     Got: '{text}'")
                        else:
                            print(
                                f"     Failed to transcribe crop {crop_counter}. Skipping."
                            )
                            # Delete the bad crop
                            os.remove(crop_filepath)

                        crop_counter += 1

                        # 4. Sleep to respect free tier rate limits
                        time.sleep(SLEEP_TIME_SECONDS)

            except Exception as e:
                print(f"[!] Error processing {img_path}: {e}")

    print("\n=======================================================")
    print(f"Done! Extracted {crop_counter} crops.")
    print(f"Dataset is ready at: {OUTPUT_DATASET_DIR}")
    print("=======================================================")


if __name__ == "__main__":
    main()
