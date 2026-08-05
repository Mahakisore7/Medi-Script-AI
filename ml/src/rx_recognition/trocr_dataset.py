import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FieldCropDataset(Dataset):
    """
    manifest_csv columns: image_path, field_type, transcription
    """

    def __init__(self, manifest_csv: str, processor):
        import os

        self.df = pd.read_csv(manifest_csv)
        self.processor = processor
        self.manifest_dir = os.path.dirname(os.path.abspath(manifest_csv))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        import os

        # Resolve image path relative to the manifest csv location
        # Replace backslashes with forward slashes in case manifest was generated on Windows
        img_path = str(row.image_path).replace("\\", "/")
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.manifest_dir, img_path)

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize the target text
        labels = self.processor.tokenizer(
            str(row.transcription), padding="max_length", max_length=32, truncation=True
        ).input_ids

        # Replace pad_token_id with -100 so cross-entropy loss ignores it
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": labels}
