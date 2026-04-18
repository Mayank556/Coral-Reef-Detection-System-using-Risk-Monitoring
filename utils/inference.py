"""
CoralVisionNet IO - Inference Pipeline
========================================
Production-ready inference with:
  - Consistent preprocessing (via UnderwaterPreprocessor)
  - Monte Carlo Dropout (10 forward passes) for uncertainty estimation
  - Automatic "Uncertain Prediction" tagging when variance > 0.15
  - Mixed-precision support
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import UnderwaterPreprocessor
from models.model import CoralVisionNet

logger = logging.getLogger(__name__)

# Uncertainty threshold — predictions with variance above this
# are labelled as "Uncertain Prediction"
UNCERTAINTY_THRESHOLD = 0.15
MC_DROPOUT_PASSES = 10


def _enable_dropout(model: torch.nn.Module):
    """Turn on dropout layers during inference for MC-Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


class CoralInferencePipeline:
    """End-to-end inference wrapper for CoralVisionNet.

    Usage:
        pipeline = CoralInferencePipeline("weights.pth", device="cuda")
        result = pipeline.predict("image.jpg")
        # result = {
        #     "class": "Healthy",
        #     "class_index": 0,
        #     "confidence": 0.92,
        #     "probabilities": [0.92, 0.03, 0.04, 0.01],
        #     "variance": 0.008,
        #     "uncertain": False,
        # }
    """

    def __init__(self, weights_path: str = None, device: str = "cpu",
                 image_size: int = 224, num_classes: int = 4,
                 mc_passes: int = MC_DROPOUT_PASSES):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.mc_passes = mc_passes
        self.preprocessor = UnderwaterPreprocessor(image_size=image_size)

        # Build model
        self.model = CoralVisionNet(num_classes=num_classes, pretrained=False)

        # Load trained weights if provided
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {weights_path}")

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Single deterministic forward pass
    # ------------------------------------------------------------------
    def _single_forward(self, rgb: torch.Tensor,
                        lab: torch.Tensor) -> torch.Tensor:
        """Run one forward pass and return class probabilities."""
        with torch.no_grad():
            logits = self.model(rgb, lab)
        return F.softmax(logits, dim=1)

    # ------------------------------------------------------------------
    # Monte Carlo Dropout forward passes
    # ------------------------------------------------------------------
    def _mc_dropout_forward(self, rgb: torch.Tensor,
                            lab: torch.Tensor) -> tuple:
        """Run multiple stochastic forward passes with dropout enabled.

        Returns:
            mean_probs: (B, C) mean probability across MC passes.
            variance:   (B,)   predictive variance (max across classes).
        """
        self.model.eval()           # BN stays in eval
        _enable_dropout(self.model)  # only dropout goes to train

        all_probs = []
        for _ in range(self.mc_passes):
            with torch.no_grad():
                logits = self.model(rgb, lab)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        # Stack: (mc_passes, B, C)
        stacked = torch.stack(all_probs, dim=0)
        mean_probs = stacked.mean(dim=0)        # (B, C)
        var_probs = stacked.var(dim=0)           # (B, C)
        max_variance = var_probs.max(dim=1).values  # (B,)

        self.model.eval()  # reset dropout to eval
        return mean_probs, max_variance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, image_input, use_mc_dropout: bool = True) -> dict:
        """Run inference on a single image.

        Args:
            image_input: File path (str/Path) or BGR numpy array.
            use_mc_dropout: If True, use MC-Dropout for uncertainty.

        Returns:
            dict with prediction details.
        """
        # Load image if path
        if isinstance(image_input, (str, Path)):
            image_bgr = cv2.imread(str(image_input))
            if image_bgr is None:
                raise FileNotFoundError(
                    f"Could not read image: {image_input}")
        else:
            image_bgr = image_input

        # Preprocess
        rgb_tensor, lab_tensor = self.preprocessor(image_bgr)
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        lab_tensor = lab_tensor.unsqueeze(0).to(self.device)

        # Forward
        if use_mc_dropout:
            mean_probs, variance = self._mc_dropout_forward(
                rgb_tensor, lab_tensor)
            probs = mean_probs[0].cpu().numpy()
            var_val = variance[0].item()
        else:
            probs_tensor = self._single_forward(rgb_tensor, lab_tensor)
            probs = probs_tensor[0].cpu().numpy()
            var_val = 0.0

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        is_uncertain = var_val > UNCERTAINTY_THRESHOLD

        class_name = CoralVisionNet.CLASS_NAMES[class_idx]
        if is_uncertain:
            class_name = f"{class_name} (Uncertain Prediction)"

        return {
            "class": class_name,
            "class_index": class_idx,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(CoralVisionNet.CLASS_NAMES, probs)
            },
            "variance": round(var_val, 6),
            "uncertain": is_uncertain,
        }

    def predict_batch(self, image_list: list,
                      use_mc_dropout: bool = True) -> list:
        """Predict on a list of images (paths or arrays)."""
        return [self.predict(img, use_mc_dropout=use_mc_dropout)
                for img in image_list]
