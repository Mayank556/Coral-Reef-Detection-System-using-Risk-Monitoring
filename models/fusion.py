"""
CoralVisionNet IO - Gated Fusion Module
========================================
Implements trainable gated fusion for combining variable feature streams.
Added temperature scaling to prevent stream collapse.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class GatedFusion(nn.Module):
    """Learnable softmax-gated fusion of feature vectors.

    The module maintains trainable scalar logits for the active streams. 
    Applies temperature scaling to soften the softmax and prevent one stream 
    from dominating too early (collapse).

    Attributes:
        gate_logits (nn.Parameter): Raw (un-normalized) gating weights.
    """

    def __init__(self, num_streams: int = 3, feature_dim: int = 512, temperature: float = 2.0, stream_names: list = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_streams = num_streams
        self.temperature = temperature
        self.stream_names = stream_names if stream_names else [f"stream_{i}" for i in range(num_streams)]

        # Learnable gating logits — initialized uniformly
        self.gate_logits = nn.Parameter(torch.zeros(num_streams))

    def forward(self, features: list) -> torch.Tensor:
        """Fuse feature vectors via softmax gating.

        Args:
            features: List of (B, 512) tensors from active streams.

        Returns:
            fused: (B, 512) weighted combination.
        """
        assert len(features) == self.num_streams, "Number of input features must match initialized num_streams."
        
        # Apply temperature scaling to prevent collapse
        weights = torch.softmax(self.gate_logits / self.temperature, dim=0)

        fused = torch.zeros_like(features[0])
        for idx, (w, f) in enumerate(zip(weights, features)):
            fused += w * f
            
        return fused

    def get_weights(self) -> dict:
        """Return current gating weights as a dictionary (detached)."""
        with torch.no_grad():
            w = torch.softmax(self.gate_logits / self.temperature, dim=0).cpu().tolist()
        
        return {name: val for name, val in zip(self.stream_names, w)}

    def log_weights(self, step: int = 0):
        """Log the current gating weights at a given training step."""
        weights = self.get_weights()
        weight_str = " | ".join([f"{k}={v:.4f}" for k, v in weights.items()])
        logger.info(f"[Step {step}] Gating weights → {weight_str}")
