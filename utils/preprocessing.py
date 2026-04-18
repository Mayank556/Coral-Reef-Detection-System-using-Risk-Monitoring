"""
CoralVisionNet IO - Underwater Image Preprocessing Pipeline
============================================================
Provides consistent preprocessing for both training and inference:
  1. Color correction (white-balance via gray-world assumption)
  2. CLAHE contrast enhancement
  3. Normalization (ImageNet stats for RGB streams, raw for LAB stream)
"""

import cv2
import numpy as np
import torch
from torchvision import transforms


# ---------------------------------------------------------------------------
# Core preprocessing functions (operate on uint8 BGR numpy arrays)
# ---------------------------------------------------------------------------

def color_correct(image: np.ndarray) -> np.ndarray:
    """Apply gray-world white-balance correction for underwater color cast.

    Args:
        image: BGR uint8 numpy array (H, W, 3).

    Returns:
        Color-corrected BGR uint8 numpy array.
    """
    img_float = image.astype(np.float32)
    channel_means = img_float.mean(axis=(0, 1))  # per-channel mean
    global_mean = channel_means.mean()

    # Scale each channel so its mean equals the global mean
    scale = global_mean / (channel_means + 1e-6)
    corrected = np.clip(img_float * scale[np.newaxis, np.newaxis, :], 0, 255)
    return corrected.astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 3.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE contrast enhancement on the L channel (LAB space).

    Args:
        image: BGR uint8 numpy array.
        clip_limit: CLAHE clip limit.
        tile_grid_size: Tile grid size for CLAHE.

    Returns:
        Contrast-enhanced BGR uint8 numpy array.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to LAB color space (float32, 0-1 range).

    Returns:
        LAB float32 numpy array of shape (H, W, 3) normalized to [0, 1].
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # L: 0-255, A: 0-255, B: 0-255  →  normalize each to [0, 1]
    lab /= 255.0
    return lab


# ---------------------------------------------------------------------------
# Unified Preprocessor
# ---------------------------------------------------------------------------

class UnderwaterPreprocessor:
    """Deterministic preprocessing pipeline for underwater coral images.

    Applies the *same* steps during training and inference, guaranteeing
    consistency. Augmentations (random flips, rotations, etc.) should be
    applied *after* this pipeline via a separate augmentation transform.

    Usage:
        preprocessor = UnderwaterPreprocessor(image_size=224)
        rgb_tensor, lab_tensor = preprocessor(bgr_image)
    """

    # ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, image_size: int = 224, clahe_clip: float = 3.0, enable_preprocessing: bool = True):
        self.image_size = image_size
        self.clahe_clip = clahe_clip
        self.enable_preprocessing = enable_preprocessing

        # Tensor conversion + ImageNet normalization for RGB streams
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 → CHW float [0,1]
            transforms.Normalize(mean=self.IMAGENET_MEAN,
                                 std=self.IMAGENET_STD),
        ])

        # Tensor conversion for LAB stream (no ImageNet norm)
        self.lab_transform = transforms.ToTensor()  # HWC float → CHW float

    def __call__(self, image_bgr: np.ndarray):
        """Process a single BGR image.

        Args:
            image_bgr: Raw BGR uint8 numpy array of any size.

        Returns:
            rgb_tensor: (3, image_size, image_size) ImageNet-normalized tensor.
            lab_tensor: (3, image_size, image_size) LAB tensor in [0, 1].
        """
        if self.enable_preprocessing:
            # 1. Color correction
            corrected = color_correct(image_bgr)
            # 2. CLAHE contrast enhancement
            enhanced = apply_clahe(corrected, clip_limit=self.clahe_clip)
        else:
            enhanced = image_bgr

        # 3. Resize
        resized = cv2.resize(enhanced, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)

        # 4a. RGB tensor for Spatial + Contextual streams
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb_tensor = self.rgb_transform(rgb_image)

        # 4b. LAB tensor for Spectral stream
        lab_image = to_lab(resized)  # float32 HWC [0,1]
        lab_tensor = self.lab_transform(lab_image)

        return rgb_tensor, lab_tensor


# ---------------------------------------------------------------------------
# Training augmentation helper (applied AFTER preprocessing)
# ---------------------------------------------------------------------------

def get_training_augmentation():
    """Return a torchvision transform for data augmentation during training.

    This should be applied to the *tensor* output of UnderwaterPreprocessor.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
    ])
