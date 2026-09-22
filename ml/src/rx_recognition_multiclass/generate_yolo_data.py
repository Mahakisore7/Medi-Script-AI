import csv
import os
import shutil

import yaml


def generate_dataset():
    ds_dir = "data/rx-fields-dataset"
    for split in ["train", "val"]:
        os.makedirs(f"{ds_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{ds_dir}/labels/{split}", exist_ok=True)

    csv_path = "data/samples/ground_truth.csv"
    if not os.path.exists(csv_path):
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # split 8 train, 2 val
    for i, row in enumerate(rows):
        split = "train" if i < 8 else "val"
        img_name = row["image_file"]
        img_src = f"data/samples/{img_name}"
        img_dst = f"{ds_dir}/images/{split}/{img_name}"
        shutil.copy(img_src, img_dst)

        # Write dummy YOLO annotations
        txt_name = img_name.replace(".png", ".txt")
        txt_dst = f"{ds_dir}/labels/{split}/{txt_name}"

        with open(txt_dst, "w") as tf:
            # Dummy normalized bbox (class_id center_x center_y width height)
            tf.write("0 0.5 0.1 0.8 0.1\n")
            tf.write("1 0.5 0.3 0.8 0.1\n")
            tf.write("2 0.5 0.5 0.8 0.1\n")
            tf.write("3 0.5 0.7 0.8 0.1\n")
            tf.write("4 0.5 0.9 0.8 0.1\n")

    yaml_dict = {
        "path": os.path.abspath(ds_dir).replace(
            chr(92), "/"
        ),  # Use absolute path with forward slashes for YOLO
        "train": "images/train",
        "val": "images/val",
        "names": {0: "name", 1: "age", 2: "gender", 3: "symptoms", 4: "medicine", 5: "dosage"},
    }
    with open("ml/rx_fields.yaml", "w") as yf:
        yaml.dump(yaml_dict, yf, default_flow_style=False)

    print("YOLO dataset generated at", ds_dir)


if __name__ == "__main__":
    generate_dataset()
