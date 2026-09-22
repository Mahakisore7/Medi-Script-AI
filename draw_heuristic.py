import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "src")))
import sys
from rx_recognition_heuristic.pipeline import _get_yolo_boxes, is_overlapping_any
from rx_recognition_heuristic.regex_classifier import classify_text_heuristic


def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = r"C:\Users\yashw_d6scpoj\Downloads\sample.jpg"

    out_path = r"C:\Users\yashw_d6scpoj\.gemini\antigravity-ide\brain\26152515-2a7c-4bb3-8183-fdd9c73a7405\annotated_sample.jpg"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_weights_path = os.path.join(
        script_dir, "runs", "detect", "yolo_real_world_best.pt"
    )

    image = cv2.imread(img_path)
    if image is None:
        return

    h_img, w_img = image.shape[:2]

    # 1. Anchor (Now a list of boxes)
    yolo_boxes = _get_yolo_boxes(image, yolo_weights_path)
    if yolo_boxes:
        for anchor_box in yolo_boxes:
            # Draw YOLO Anchor in RED
            cv2.rectangle(
                image,
                (anchor_box[0], anchor_box[1]),
                (anchor_box[2], anchor_box[3]),
                (0, 0, 255),
                3,
            )
            cv2.putText(
                image,
                "YOLO Crop",
                (anchor_box[0], max(0, anchor_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    # 2. EasyOCR
    import easyocr
    import torch

    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=use_gpu)
    ocr_results = reader.readtext(image, detail=1, paragraph=True)

    for result in ocr_results:
        if len(result) == 2:
            bbox, text = result
        else:
            bbox, text, conf = result
            if conf * 100 < 20.0:
                continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        text_box = [x1, y1, x2, y2]

        if yolo_boxes and is_overlapping_any(text_box, yolo_boxes, threshold=0.1):
            # Inside/Overlapping Anchor -> GREEN
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                "Skipped",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        else:
            # Outside -> Classify and draw in BLUE
            y_center_normalized = ((y1 + y2) / 2) / h_img
            label = classify_text_heuristic(text, y_center_normalized)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    cv2.imwrite(out_path, image)
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
