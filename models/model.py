"""
CoralVisionNet IO - Tri-Stream Model Architecture
===================================================
Three parallel feature extraction streams, each producing a 512-dim vector:
  1. Spatial Stream   — ResNet50 (pretrained, fine-tuned)
  2. Contextual Stream — ViT-B/16 (pretrained, fine-tuned)
  3. Spectral Stream  — Custom CNN (SpectralNet) on LAB images

Streams are combined via GatedFusion → MLP classification head → 4 classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .fusion import GatedFusion


# ===========================================================================
# 1. Spatial Stream (ResNet50 → 512-dim)
# ===========================================================================

class SpatialStream(nn.Module):
    """ResNet50 backbone with projection to 512-dim feature vector."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Remove the original FC layer; resnet50 outputs 2048-dim
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, 3, 224, 224) RGB tensor.  Returns: (B, 512)."""
        features = self.backbone(x)
        return self.projection(features)


# ===========================================================================
# 2. Contextual Stream (ViT-B/16 → 512-dim)
# ===========================================================================

class ContextualStream(nn.Module):
    """Vision Transformer B/16 backbone with projection to 512-dim."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
        )
        # ViT's default head outputs 1000; we replace it with identity
        # and capture the 768-dim CLS token embedding.
        self.backbone = vit
        self.backbone.heads = nn.Identity()  # raw 768-dim CLS token

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, 3, 224, 224) RGB tensor.  Returns: (B, 512)."""
        cls_token = self.backbone(x)  # (B, 768)
        return self.projection(cls_token)


# ===========================================================================
# 3. Spectral Stream (SpectralNet — CNN on LAB images → 512-dim)
# ===========================================================================

class SpectralNet(nn.Module):
    """Lightweight CNN operating on LAB color space images.

    Replaces the old heuristic HSV rules with a learned spectral analyser.
    Architecture: 4 conv blocks → global avg pool → FC 512.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (B, 256, 1, 1)
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, 3, 224, 224) LAB tensor [0,1].  Returns: (B, 512)."""
        features = self.features(x)
        return self.projection(features)


# ===========================================================================
# 4. Classification Head (MLP with BN + Dropout)
# ===========================================================================

class ClassificationHead(nn.Module):
    """MLP classification head: 512 → 256 → 128 → num_classes.

    Includes batch normalization and dropout for regularization.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, 512).  Returns: (B, num_classes) raw logits."""
        return self.head(x)


# ===========================================================================
# 5. Full CoralVisionNet Model
# ===========================================================================

class CoralVisionNet(nn.Module):
    """Tri-stream hybrid network for coral reef health classification.

    Streams:
        - SpatialStream (ResNet50)   → 512-dim
        - ContextualStream (ViT-B/16) → 512-dim
        - SpectralNet (LAB CNN)       → 512-dim

    Fusion:
        GatedFusion (softmax α + β + γ = 1)

    Head:
        MLP  512 → 256 → 128 → 4
    """

    CLASS_NAMES = ["Bleached", "Dead", "Healthy", "PartiallyBleached"]

    def __init__(self, num_classes: int = 4, pretrained: bool = True,
                 dropout: float = 0.3, mode: str = "full"):
        super().__init__()
        self.mode = mode
        
        # Conditionally initialize streams based on ablation mode
        self.spatial_stream = SpatialStream(pretrained=pretrained) if mode in ["full", "resnet_only", "resnet_spectral"] else None
        self.contextual_stream = ContextualStream(pretrained=pretrained) if mode in ["full", "vit_only"] else None
        self.spectral_stream = SpectralNet() if mode in ["full", "resnet_spectral"] else None
        
        active_streams = []
        if self.spatial_stream: active_streams.append("spatial")
        if self.contextual_stream: active_streams.append("contextual")
        if self.spectral_stream: active_streams.append("spectral")
        
        num_streams = len(active_streams)
        if num_streams > 1:
            self.fusion = GatedFusion(num_streams=num_streams, feature_dim=512, stream_names=active_streams)
        else:
            self.fusion = None
            
        self.classifier = ClassificationHead(num_classes=num_classes,
                                             dropout=dropout)

    def forward(self, rgb: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """Forward pass through all three streams.

        Args:
            rgb: (B, 3, 224, 224) ImageNet-normalized RGB tensor.
            lab: (B, 3, 224, 224) LAB tensor in [0, 1].

        Returns:
            logits: (B, num_classes) raw class logits.
        """
        features = []
        if self.spatial_stream: features.append(self.spatial_stream(rgb))
        if self.contextual_stream: features.append(self.contextual_stream(rgb))
        if self.spectral_stream: features.append(self.spectral_stream(lab))

        if len(features) > 1:
            fused = self.fusion(features)
        else:
            fused = features[0]
            
        logits = self.classifier(fused)
        return logits

    def get_stream_features(self, rgb: torch.Tensor,
                            lab: torch.Tensor) -> dict:
        """Extract intermediate features for XAI / analysis.

        Returns:
            dict with keys: 'spatial', 'contextual', 'spectral', 'fused',
            each mapping to a (B, 512) tensor.
        """
        f_dict = {}
        features = []
        
        if self.spatial_stream:
            f_s = self.spatial_stream(rgb)
            f_dict["spatial"] = f_s
            features.append(f_s)
            
        if self.contextual_stream:
            f_c = self.contextual_stream(rgb)
            f_dict["contextual"] = f_c
            features.append(f_c)
            
        if self.spectral_stream:
            f_sp = self.spectral_stream(lab)
            f_dict["spectral"] = f_sp
            features.append(f_sp)

        if len(features) > 1:
            f_fused = self.fusion(features)
        else:
            f_fused = features[0]
            
        f_dict["fused"] = f_fused
        return f_dict
