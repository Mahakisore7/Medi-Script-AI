import numpy as np

from mediscript.data_gen.augment import apply_transform, get_train_transforms


def test_get_train_transforms():
    """Ensure the albumentations pipeline initializes correctly."""
    transform = get_train_transforms()
    assert transform is not None


def test_apply_transform():
    """Ensure we can apply transformations to a dummy image without errors."""
    # Create a dummy image (H=100, W=200, C=3) with random pixel values (0-255)
    dummy_image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

    # Apply training transformation
    augmented = apply_transform(dummy_image, is_training=True)

    # Assert shape and type remain correct
    assert augmented.shape == (100, 200, 3)
    assert augmented.dtype == np.uint8

    # Apply eval transformation
    eval_augmented = apply_transform(dummy_image, is_training=False)
    assert eval_augmented.shape == (100, 200, 3)
