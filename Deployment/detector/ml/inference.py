"""
Model loading and inference helpers.

Usage
-----
model = load_model(weights_path, model_type='3dcnn', device='cpu')
result = predict(model, tensor, device)
# → {'label': 'shoplifting', 'confidence': 0.91}
"""

import torch
import torch.nn.functional as F

from .model import ThreeDCNN, CNNRNN, VideoTransformer, PretrainedR3D

LABELS = {0: "non_shoplifting", 1: "shoplifting"}


# ─────────────────────────────────────────────────────────────────────────────
def load_model(weights_path: str, model_type: str = "3dcnn",
               device: str = "cpu") -> torch.nn.Module:
    """
    Instantiate the model architecture and load trained weights.

    Parameters
    ----------
    weights_path : str  – path to the .pth file
    model_type   : str  – '3dcnn' | 'cnnrnn' | 'transformer'
    device       : str  – 'cpu' | 'cuda'
    """
    model_type = model_type.lower()

    if model_type == "3dcnn":
        model = ThreeDCNN(num_classes=2)
    elif model_type == "cnnrnn":
        model = CNNRNN(num_classes=2, hidden_size=128, num_layers=2)
    elif model_type == "transformer":
        model = VideoTransformer(num_classes=2, d_model=512, nhead=8, num_layers=4)
    elif model_type == "pretrained_r3d":
        model = PretrainedR3D(num_classes=2)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         "Choose '3dcnn', 'cnnrnn', 'transformer', or 'pretrained_r3d'.")

    state = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
        
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
def predict(model: torch.nn.Module,
            tensor: torch.Tensor,
            device: str = "cpu") -> dict:
    """
    Run inference and return a human-readable result dict.

    Parameters
    ----------
    model  : loaded PyTorch model (already in eval mode)
    tensor : preprocessed video tensor (batch_size=1)
    device : computation device string

    Returns
    -------
    dict with keys:
        label      – 'shoplifting' or 'non_shoplifting'
        confidence – float in [0, 1]
        scores     – {label: probability, ...}
    """
    tensor = tensor.to(device)

    # video_utils load_video_frames returns (B, C, T, H, W)
    # ThreeDCNN expects (B, C, T, H, W)
    # Other models (CNNRNN, VideoTransformer, PretrainedR3D) expect (B, T, C, H, W)
    from .model import ThreeDCNN
    if not isinstance(model, ThreeDCNN):
        tensor = tensor.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        logits = model(tensor)                         # (1, 2)
        probs  = F.softmax(logits, dim=1).squeeze(0)   # (2,)

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    label      = LABELS[pred_idx]

    scores = {LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))}

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "scores":     scores,
    }
