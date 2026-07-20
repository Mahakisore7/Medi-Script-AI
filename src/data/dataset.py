import torch
from torch.utils.data import Dataset
from PIL import Image
from src.data.preprocessing import preprocess_image

class HandwritingDataset(Dataset):
    """
    A PyTorch Dataset for handwritten text images.
    Takes in a list of image paths, a list of text labels, and a TrOCRProcessor.
    """
    def __init__(self, df, processor, max_target_length=128, apply_preprocessing=True):
        """
        Args:
            df (pd.DataFrame): Dataframe with 'file_name' and 'text' columns.
            processor (TrOCRProcessor): HuggingFace processor for TrOCR.
            max_target_length (int): Max token length for the labels.
            apply_preprocessing (bool): Whether to apply custom OpenCV preprocessing.
        """
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.apply_preprocessing = apply_preprocessing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get file path and text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        # 2. Load and preprocess image
        try:
            if self.apply_preprocessing:
                image = preprocess_image(file_name)
            else:
                image = Image.open(file_name).convert("RGB")
        except Exception as e:
            # Fallback for broken images, returning a black image
            print(f"Warning: Failed to load {file_name}: {e}")
            image = Image.new('RGB', (384, 384), (0, 0, 0))
            text = ""

        # 3. Use processor to create pixel values
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # 4. Use processor to create token labels
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # HuggingFace expects labels of -100 to ignore for loss computation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
        
        return encoding
