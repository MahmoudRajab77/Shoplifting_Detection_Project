"""
Video preprocessing utilities for inference.
Extracts N evenly-spaced frames from a video file and returns
a normalised float32 tensor ready to feed into the model.
"""

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
def load_video_frames(video_path: str, num_frames: int = 20) -> torch.Tensor:
    """
    Load `num_frames` evenly-spaced frames from *video_path*.

    Returns
    -------
    torch.Tensor  shape (1, C, T, H, W)  for ThreeDCNN
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video has no readable frames.")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

    cap.release()

    # Pad with last frame if some reads failed
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), np.uint8))

    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
    frames = frames.transpose(3, 0, 1, 2)                          # (C, T, H, W)

    # Add batch dim → (1, C, T, H, W)
    return torch.tensor(frames).unsqueeze(0)
