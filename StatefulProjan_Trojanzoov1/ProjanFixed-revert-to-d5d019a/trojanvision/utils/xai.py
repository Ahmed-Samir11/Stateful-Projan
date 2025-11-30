import os
import torch
import torch.nn.functional as F
from typing import Optional

_HAS_CAPTUM = False
try:
    from captum.attr import IntegratedGradients, LayerGradCam
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False


def _normalize_attribution(attr: torch.Tensor) -> torch.Tensor:
    # Normalize to [0,1] for visualization
    attr = attr.detach()
    attr = attr - attr.min()
    denom = (attr.max() - attr.min() + 1e-8)
    return attr / denom


def integrated_gradients(model, inputs: torch.Tensor, target_class: int, baseline: Optional[torch.Tensor] = None, steps: int = 32) -> torch.Tensor:
    """Compute Integrated Gradients over the input. Returns attribution map with shape like inputs."""
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    if _HAS_CAPTUM:
        ig = IntegratedGradients(model)
        attributions = ig.attribute(inputs, baseline, target=target_class, n_steps=steps)
    else:
        # Fallback: simple input gradient as a rough proxy
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, torch.full((inputs.size(0),), target_class, dtype=torch.long, device=inputs.device))
        loss.backward()
        attributions = inputs.grad
    return _normalize_attribution(attributions.abs().sum(dim=1, keepdim=True))  # sum over channels


def grad_cam(model, inputs: torch.Tensor, target_layer: torch.nn.Module, target_class: int) -> torch.Tensor:
    """Compute (Layer) Grad-CAM heatmaps for a given layer; requires captum. Fallback to zeros if unavailable."""
    model.eval()
    if not _HAS_CAPTUM:
        return torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3)), device=inputs.device)
    lgc = LayerGradCam(model, target_layer)
    cam = lgc.attribute(inputs, target=target_class)
    # Upsample to input size if needed
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
    return _normalize_attribution(cam)


def save_heatmaps(heatmaps: torch.Tensor, filename_prefix: str, out_dir: str) -> None:
    """Save heatmaps as image grids (requires matplotlib). Stored as PNGs."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        os.makedirs(out_dir, exist_ok=True)
        b = min(heatmaps.size(0), 8)
        h = heatmaps[:b].detach().cpu().squeeze(1).numpy()  # (b, H, W)
        
        fig, axs = plt.subplots(1, b, figsize=(3*b, 3))
        if b == 1:
            axs = [axs]  # Make it iterable when only one subplot
        
        for i in range(b):
            axs[i].imshow(h[i], cmap='jet')
            axs[i].axis('off')
        
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{filename_prefix}.png")
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
    except ImportError as exc:
        print(f"[XAI] WARNING: Cannot save heatmaps - matplotlib not available: {exc}")
    except Exception as exc:
        print(f"[XAI] ERROR saving heatmap {filename_prefix}: {exc}")
        import traceback
        traceback.print_exc()