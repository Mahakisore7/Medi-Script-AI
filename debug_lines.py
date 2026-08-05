import sys
import os
import cv2

# Add ml/src to pythonpath so imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "ml", "src"))

from rx_recognition.segmentation import segment_fields

image_path = r"C:\Users\yashw_d6scpoj\Desktop\Academics\3rd Year\Sem 5\Deep Learning\Project\Medi-Script-AI\my sample\original_prescription_1.jpg"
image = cv2.imread(image_path)

print("Segmenting...")
fields = segment_fields(image, mode="text")

print("Recognizing lines...")
from rx_recognition.recognition import TrOCRRecognizer

recognizer = TrOCRRecognizer("microsoft/trocr-base-handwritten")

for i, (field_name, crop, seg_conf) in enumerate(fields):
    raw_text, _ = recognizer.recognize_text(crop)
    print(f"{field_name}: {raw_text}")
