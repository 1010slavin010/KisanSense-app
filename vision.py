"""
vision.py — lightweight placeholder crop-health heuristic.

Swap analyze_image() for a real trained edge-AI model (e.g. a small CNN
flashed to the hub) once it exists. For now this uses a simple green-pixel
ratio heuristic on the uploaded photo so the UI has something real to show.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class VisionResult:
    verdict: str
    green_ratio: float
    note: str


def analyze_image(image: Image.Image) -> VisionResult:
    img = image.convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype("float32")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # A pixel counts as "green/healthy canopy" if green channel dominates.
    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 40)
    green_ratio = float(green_mask.mean())

    if green_ratio > 0.55:
        verdict, note = "Healthy", "Canopy looks predominantly green — no obvious stress detected."
    elif green_ratio > 0.3:
        verdict, note = "Fair", "Some green cover, but patchiness or discoloration is visible — keep monitoring."
    else:
        verdict, note = "Needs attention", "Low green cover detected — check for stress, disease, or pests."

    return VisionResult(verdict=verdict, green_ratio=green_ratio, note=note)
