import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FieldCropDataset(Dataset):
    """
    manifest_csv columns: image_path, field_type, transcription
    """

    def __init__(self, manifest_csv: str, processor, is_train: bool = False):
        import os

        self.df = pd.read_csv(manifest_csv)
        self.processor = processor
        self.manifest_dir = os.path.dirname(os.path.abspath(manifest_csv))
        self.is_train = is_train

        if self.is_train:
            self.transform = T.Compose([
                T.RandomRotation(degrees=3, expand=True, fill=255),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
            ])
        else:
            self.transform = None

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

        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize the target text
        labels = self.processor.tokenizer(
            str(row.transcription), padding="max_length", max_length=32, truncation=True
        ).input_ids

        # Replace pad_token_id with -100 so cross-entropy loss ignores it
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": labels}
