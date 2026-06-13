"""
FastViT Model - Cytological Pap Smear Image Classification
Implements FastViT-T8 with GradCAM explainability for SipakMed categories

SipakMed Cell Categories:
  0: Dyskeratotic    - Abnormal superficial/intermediate cells
  1: Koilocytotic    - HPV-infected cells  
  2: Metaplastic     - Immature squamous metaplasia
  3: Parabasal       - Basal layer cells (atrophic)
  4: Superficial-Intermediate - Normal cells
"""

import os
import io
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import base64

logger = logging.getLogger("CervicalAI.FastViT")

# SipakMed Class definitions
SIPAKMED_CLASSES = {
    0: {"name": "Dyskeratotic", "risk": "high", "description": "Abnormal cells with premature keratinization. Associated with squamous cell carcinoma."},
    1: {"name": "Koilocytotic", "risk": "high", "description": "HPV-infected cells showing perinuclear halo. Strong indicator of HPV infection."},
    2: {"name": "Metaplastic", "risk": "medium", "description": "Squamous metaplasia cells. May indicate cervical transformation zone activity."},
    3: {"name": "Parabasal", "risk": "medium", "description": "Small basal layer cells. Common in atrophic or post-menopausal cervix."},
    4: {"name": "Superficial-Intermediate", "risk": "low", "description": "Normal superficial and intermediate squamous cells. Healthy cervical epithelium."}
}

RISK_LEVEL_SCORES = {"low": 0.15, "medium": 0.45, "high": 0.85}

# Image preprocessing pipeline (ImageNet normalization for transfer learning)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────────────────────
# FastViT Architecture (Lightweight ViT approximation using
# depthwise separable convolutions + token mixing)
# ─────────────────────────────────────────────────────────────

class RepConv(nn.Module):
    """Re-parameterizable convolution block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FastViTBlock(nn.Module):
    """FastViT building block with token mixing via large kernel depthwise conv."""
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        # Token mixing (spatial)
        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mix = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False),  # Depthwise
            nn.Conv2d(dim, dim, 1, bias=False),  # Pointwise
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        # Channel MLP
        self.norm2 = nn.BatchNorm2d(dim)
        mid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mid, 1),
            nn.GELU(),
            nn.Dropout2d(drop),
            nn.Conv2d(mid, dim, 1),
        )
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        x = x + self.drop(self.token_mix(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Stem: aggressive downsampling via stacked convolutions."""
    def __init__(self, in_ch=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            RepConv(in_ch, 32, 3, 2),
            RepConv(32, 64, 3, 2),
            RepConv(64, embed_dim, 3, 2),
        )

    def forward(self, x):
        return self.proj(x)


class FastViTT8(nn.Module):
    """
    FastViT-T8 approximation for cytological image classification.
    Architecture: 4 stages with increasing channel dimensions.
    Total params: ~6M (suitable for medical imaging with limited data)
    """
    def __init__(self, num_classes=5, drop_rate=0.1):
        super().__init__()
        # Stage dimensions: [64, 128, 256, 512]
        dims = [64, 128, 256, 512]

        # Stem
        self.stem = PatchEmbed(3, dims[0])

        # Stage 1
        self.stage1 = nn.Sequential(*[FastViTBlock(dims[0]) for _ in range(2)])
        self.down1 = RepConv(dims[0], dims[1], 3, 2)

        # Stage 2
        self.stage2 = nn.Sequential(*[FastViTBlock(dims[1]) for _ in range(2)])
        self.down2 = RepConv(dims[1], dims[2], 3, 2)

        # Stage 3
        self.stage3 = nn.Sequential(*[FastViTBlock(dims[2]) for _ in range(2)])
        self.down3 = RepConv(dims[2], dims[3], 3, 2)

        # Stage 4 (deepest features - used for GradCAM)
        self.stage4 = nn.Sequential(*[FastViTBlock(dims[3]) for _ in range(2)])

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[3]),
            nn.Dropout(drop_rate),
            nn.Linear(dims[3], num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────
# GradCAM Implementation
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for FastViT.
    Generates saliency heatmaps showing which regions drove the prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        score = output[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # [C]

        # Weight activations
        cam = self.activations[0]  # [C, H, W]
        for i, w in enumerate(pooled_grads):
            cam[i] *= w

        cam = F.relu(cam.mean(0))  # [H, W]
        cam = cam.numpy()

        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# ─────────────────────────────────────────────────────────────
# Model Management (singleton with lazy loading)
# ─────────────────────────────────────────────────────────────

_model_instance: Optional[FastViTT8] = None
MODEL_PATH = Path(os.getenv("MODEL_DIR", "models")) / "fastvit_sipakmed.pt"


def get_fastvit_model() -> FastViTT8:
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    device = torch.device("cpu")  # CPU for inference on Render free tier
    model = FastViTT8(num_classes=5)

    if MODEL_PATH.exists():
        logger.info(f"Loading FastViT weights from {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info("✅ FastViT weights loaded")
    else:
        logger.warning(f"⚠️ No saved weights at {MODEL_PATH}. Using randomly initialized model.")
        logger.warning("Run training script or place pretrained weights at models/fastvit_sipakmed.pt")

    model.eval()
    _model_instance = model
    return model


# ─────────────────────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess image bytes into model-ready tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    tensor = TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor, img_np


def generate_gradcam_overlay(
    model: FastViTT8,
    tensor: torch.Tensor,
    original_img: np.ndarray,
    pred_class: int
) -> str:
    """
    Generate GradCAM heatmap overlaid on original image.
    Returns base64-encoded PNG string.
    """
    # Target the last block of stage4 for GradCAM
    target_layer = model.stage4[-1].token_mix[0]  # Depthwise conv in last block
    gradcam = GradCAM(model, target_layer)

    try:
        cam = gradcam(tensor.clone().requires_grad_(True), pred_class)
        gradcam.remove_hooks()

        # Resize CAM to image size
        h, w = original_img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        orig_resized = cv2.resize(original_img, (w, h))
        overlay = (0.55 * orig_resized + 0.45 * heatmap).astype(np.uint8)

        # Encode to base64
        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        logger.error(f"GradCAM generation failed: {e}")
        # Return original image as fallback
        pil_img = Image.fromarray(original_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"


def predict_image(image_bytes: bytes) -> Dict:
    """
    Full inference pipeline: preprocess → classify → GradCAM.
    Returns structured prediction result.
    """
    model = get_fastvit_model()
    tensor, img_np = preprocess_image(image_bytes)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].numpy()

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])
    cell_info = SIPAKMED_CLASSES[pred_class]

    # Generate GradCAM
    gradcam_img = generate_gradcam_overlay(model, tensor, img_np, pred_class)

    # Build per-class probabilities
    class_probs = [
        {
            "class_id": i,
            "class_name": SIPAKMED_CLASSES[i]["name"],
            "probability": float(probs[i]),
            "risk": SIPAKMED_CLASSES[i]["risk"]
        }
        for i in range(5)
    ]

    return {
        "predicted_class": pred_class,
        "class_name": cell_info["name"],
        "confidence": confidence,
        "risk_level": cell_info["risk"],
        "risk_score": RISK_LEVEL_SCORES[cell_info["risk"]],
        "description": cell_info["description"],
        "class_probabilities": class_probs,
        "gradcam_image": gradcam_img,
        "model": "FastViT-T8",
        "dataset": "SipakMed"
    }
