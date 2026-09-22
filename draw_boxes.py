import cv2
import os

img_path = "input_images/104.jpg"
label_path = "output_dataset/labels/104.txt"
out_path = r"C:\Users\yashw_d6scpoj\.gemini\antigravity-ide\brain\26152515-2a7c-4bb3-8183-fdd9c73a7405\annotated_104.jpg"

if not os.path.exists(img_path):
    # Try finding the correct file extension if it's png or something
    print(f"Error: {img_path} not found.")
    exit(1)

img = cv2.imread(img_path)
h, w, _ = img.shape

classes = ["Diagnosis", "DocterName", "Doctor Sign", "Medical Text", "PatientInfo"]
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]

with open(label_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        x_c, y_c, box_w, box_h = map(float, parts[1:])

        x1 = int((x_c - box_w / 2) * w)
        y1 = int((y_c - box_h / 2) * h)
        x2 = int((x_c + box_w / 2) * w)
        y2 = int((y_c + box_h / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls_id], 2)
        cv2.putText(
            img,
            classes[cls_id],
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[cls_id],
            2,
        )

cv2.imwrite(out_path, img)
print("Saved to", out_path)
