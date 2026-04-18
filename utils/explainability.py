"""
CoralVisionNet IO - Unified Explainability Module
===================================================
Combines three XAI techniques into a single overlay heatmap:
  1. Grad-CAM++ for the CNN-based streams (Spatial / Spectral)
  2. Attention Rollout for the ViT Contextual stream
  3. Spectral activation maps from SpectralNet intermediate layers

All maps are resized, normalized, and alpha-blended into one unified
heatmap overlaid on the original image.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


# ===================================================================
# 1. Grad-CAM++ (for ResNet50 / SpectralNet conv layers)
# ===================================================================

class GradCAMPlusPlus:
    """Grad-CAM++ for any convolutional backbone.

    Usage:
        gcam = GradCAMPlusPlus(model.spatial_stream.backbone,
                               target_layer=model.spatial_stream.backbone[-2])
        heatmap = gcam.generate(input_tensor, class_idx=0)
    """

    def __init__(self, model: torch.nn.Module,
                 target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(
            self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(
            self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int) -> np.ndarray:
        """Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: (1, C, H, W) input to the model.
            class_idx: Target class index.

        Returns:
            heatmap: (H, W) numpy float32 array, values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        # Handle both raw feature and logit outputs
        if output.dim() == 2:
            score = output[0, class_idx]
        else:
            score = output.view(-1)[class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self._gradients  # (1, C, h, w)
        acts = self._activations  # (1, C, h, w)

        # Grad-CAM++ weighting
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True) + 1e-7

        alpha = grads_power_2 / (
            2 * grads_power_2 + sum_acts * grads_power_3 + 1e-7
        )
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    def release(self):
        """Remove registered hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ===================================================================
# 2. Attention Rollout (for ViT)
# ===================================================================

def attention_rollout(model_vit: torch.nn.Module,
                      input_tensor: torch.Tensor) -> np.ndarray:
    """Compute attention rollout for a torchvision ViT model.

    This extracts attention weights from every encoder block and
    recursively multiplies them to produce a global attention map
    from the CLS token to all spatial patches.

    Args:
        model_vit: A torchvision ViT model (models.vit_b_16).
        input_tensor: (1, 3, 224, 224) RGB tensor.

    Returns:
        attention_map: (H_patches, W_patches) float32 array in [0, 1].
    """
    attention_weights = []

    # Hook to capture attention weights
    hooks = []
    for block in model_vit.encoder.layers:
        def hook_fn(module, input, output, _block=block):
            # torchvision ViT's self_attention returns (attn_output,
            # attn_weights) only when need_weights=True. We'll use a
            # workaround: register on the full block and compute attn
            # manually from q,k.
            pass
        hooks.append(block.register_forward_hook(hook_fn))

    # Alternative: compute attention from Q, K directly
    # We patch the self_attention forward temporarily
    original_fns = []
    for block in model_vit.encoder.layers:
        sa = block.self_attention
        original_fn = sa.forward

        def make_patched(orig_sa, orig_forward):
            def patched_forward(*args, **kwargs):
                # Get Q, K from input
                x = args[0] if args else kwargs.get("query", None)
                if x is None:
                    return orig_forward(*args, **kwargs)

                B, N, C = x.shape
                num_heads = orig_sa.num_heads
                head_dim = C // num_heads

                # Compute Q, K via in_proj
                qkv = F.linear(x, orig_sa.in_proj_weight,
                                orig_sa.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)
                k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)

                attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                attn = attn.softmax(dim=-1)

                # Average over heads → (B, N, N)
                attn_avg = attn.mean(dim=1)
                attention_weights.append(attn_avg.detach().cpu())

                return orig_forward(*args, **kwargs)
            return patched_forward

        sa.forward = make_patched(sa, original_fn)
        original_fns.append((sa, original_fn))

    # Forward pass
    with torch.no_grad():
        model_vit(input_tensor)

    # Restore original forward functions
    for sa, orig_fn in original_fns:
        sa.forward = orig_fn
    for h in hooks:
        h.remove()

    if not attention_weights:
        # Fallback: uniform attention
        return np.ones((14, 14), dtype=np.float32) / (14 * 14)

    # Rollout: recursively multiply attention matrices
    result = attention_weights[0].squeeze(0).numpy()  # (N, N)
    for attn in attention_weights[1:]:
        attn_np = attn.squeeze(0).numpy()
        # Add identity (residual connection)
        attn_np = 0.5 * attn_np + 0.5 * np.eye(attn_np.shape[0])
        result = attn_np @ result

    # CLS token attention to patches (skip CLS-to-CLS at index 0)
    cls_attention = result[0, 1:]  # (num_patches,)
    num_patches = int(cls_attention.shape[0] ** 0.5)
    attn_map = cls_attention.reshape(num_patches, num_patches)

    # Normalize
    attn_map = attn_map - attn_map.min()
    if attn_map.max() > 0:
        attn_map = attn_map / attn_map.max()

    return attn_map.astype(np.float32)


# ===================================================================
# 3. Spectral Activation Map (from SpectralNet intermediate features)
# ===================================================================

def spectral_activation_map(spectral_stream: torch.nn.Module,
                             lab_tensor: torch.Tensor) -> np.ndarray:
    """Extract and visualize activation from the last conv block
    of SpectralNet.

    Args:
        spectral_stream: The SpectralNet module.
        lab_tensor: (1, 3, 224, 224) LAB tensor.

    Returns:
        activation_map: (h, w) float32 array in [0, 1].
    """
    activation = None

    # Hook onto the last conv layer (before AdaptiveAvgPool2d)
    # In SpectralNet, index [-2] is the ReLU before the pool
    target_layer = None
    for i, layer in enumerate(spectral_stream.features):
        if isinstance(layer, torch.nn.Conv2d):
            target_layer = layer  # last conv2d

    if target_layer is None:
        return np.ones((14, 14), dtype=np.float32) / (14 * 14)

    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        spectral_stream(lab_tensor)

    handle.remove()

    if activation is None:
        return np.ones((14, 14), dtype=np.float32) / (14 * 14)

    # Average across channels → (h, w)
    act_map = activation.squeeze(0).mean(dim=0).cpu().numpy()
    act_map = act_map - act_map.min()
    if act_map.max() > 0:
        act_map = act_map / act_map.max()

    return act_map.astype(np.float32)


# ===================================================================
# 4. Unified Heatmap Overlay
# ===================================================================

class UnifiedXAI:
    """Unified explainability combining all three stream visualizations.

    Usage:
        xai = UnifiedXAI(model)
        overlay, maps = xai.explain(rgb_tensor, lab_tensor, class_idx=0,
                                     original_image=bgr_image)
    """

    def __init__(self, model):
        """
        Args:
            model: CoralVisionNet instance.
        """
        self.model = model

        # Grad-CAM++ for Spatial stream (target: last bottleneck block)
        spatial_backbone = model.spatial_stream.backbone
        # Last conv layer in ResNet50 backbone is layer4 (index -2 in
        # the Sequential since we removed avgpool and fc)
        self.grad_cam_spatial = GradCAMPlusPlus(
            model=model.spatial_stream,
            target_layer=spatial_backbone[-2]  # layer4
        )

    def explain(self, rgb_tensor: torch.Tensor,
                lab_tensor: torch.Tensor,
                class_idx: int,
                original_image: Optional[np.ndarray] = None,
                image_size: int = 224,
                weights: tuple = (0.4, 0.35, 0.25)) -> tuple:
        """Generate unified XAI heatmap.

        Args:
            rgb_tensor: (1, 3, 224, 224) RGB tensor.
            lab_tensor: (1, 3, 224, 224) LAB tensor.
            class_idx: Target class index for explanation.
            original_image: Optional BGR image for overlay.
            image_size: Output size for the heatmap.
            weights: (w_spatial, w_contextual, w_spectral) blending weights
                     for the three attention maps.

        Returns:
            overlay: BGR uint8 image with heatmap overlay (or None).
            maps: dict with individual heatmaps.
        """
        rgb_tensor = rgb_tensor.requires_grad_(True)
        lab_tensor_grad = lab_tensor.clone().requires_grad_(True)

        # 1. Grad-CAM++ for Spatial stream
        spatial_map = self.grad_cam_spatial.generate(rgb_tensor, class_idx)

        # 2. Attention Rollout for Contextual stream (ViT)
        vit_backbone = self.model.contextual_stream.backbone
        context_map = attention_rollout(vit_backbone, rgb_tensor.detach())

        # 3. Spectral Activation Map
        spec_map = spectral_activation_map(
            self.model.spectral_stream, lab_tensor_grad.detach())

        # Resize all maps to the same size
        spatial_resized = cv2.resize(spatial_map, (image_size, image_size))
        context_resized = cv2.resize(context_map, (image_size, image_size))
        spec_resized = cv2.resize(spec_map, (image_size, image_size))

        # Explicitly min-max normalize maps before fusion to ensure equality
        def min_max_norm(m):
            m = m - m.min()
            return m / m.max() if m.max() > 0 else m

        spatial_resized = min_max_norm(spatial_resized)
        context_resized = min_max_norm(context_resized)
        spec_resized = min_max_norm(spec_resized)

        # Weighted combination
        w_s, w_c, w_sp = weights
        unified = (w_s * spatial_resized +
                   w_c * context_resized +
                   w_sp * spec_resized)

        # Normalize unified map
        unified = unified - unified.min()
        if unified.max() > 0:
            unified = unified / unified.max()

        # Create colored heatmap
        heatmap_color = cv2.applyColorMap(
            (unified * 255).astype(np.uint8), cv2.COLORMAP_JET)

        overlay = None
        if original_image is not None:
            resized_orig = cv2.resize(original_image,
                                      (image_size, image_size))
            overlay = cv2.addWeighted(resized_orig, 0.6,
                                      heatmap_color, 0.4, 0)

        maps = {
            "spatial_gradcam": spatial_resized,
            "contextual_attention": context_resized,
            "spectral_activation": spec_resized,
            "unified": unified,
            "heatmap_color": heatmap_color,
        }

        return overlay, maps

    def release(self):
        """Clean up hooks."""
        self.grad_cam_spatial.release()
