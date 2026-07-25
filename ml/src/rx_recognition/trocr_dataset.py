import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FieldCropDataset(Dataset):
    """
    manifest_csv columns: image_path, field_type, transcription
    """

    def __init__(self, manifest_csv: str, processor):
        self.df = pd.read_csv(manifest_csv)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        import os

        # resolve relative to ml/ directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        img_path = os.path.join(base_dir, "ml", row.image_path)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize the target text
        labels = self.processor.tokenizer(
            str(row.transcription), padding="max_length", max_length=32, truncation=True
        ).input_ids

        # Replace pad_token_id with -100 so cross-entropy loss ignores it
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": labels}
