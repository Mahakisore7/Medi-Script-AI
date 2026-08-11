import albumentations as A
import numpy as np


def get_train_transforms():
    """
    Returns the albumentations composed transform pipeline for training.
    This replicates the literature review's augmentation strategy (Rotation, Shift, Noise)
    but executes entirely in memory to save disk space.
    """
    return A.Compose(
        [
            A.Rotate(limit=5, p=0.5),  # mirrors "Rotate" from Tabassum et al.
            A.Affine(translate_percent=0.02, p=0.5),  # mirrors "Shift"
            A.ElasticTransform(alpha=1, sigma=15, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.3),
        ]
    )


def get_eval_transforms():
    """
    Returns the albumentations composed transform pipeline for evaluation.
    Usually just basic normalization or resizing, keeping it empty for now
    until we lock in the model's exact input resolution.
    """
    return A.Compose([])


def apply_transform(image: np.ndarray, is_training: bool = True) -> np.ndarray:
    """
    Applies the appropriate transform to a numpy image array (H, W, C).
    """
    transform = get_train_transforms() if is_training else get_eval_transforms()
    augmented = transform(image=image)
    return augmented["image"]
