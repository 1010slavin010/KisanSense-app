"""
Very lightweight stand-in for the hub's edge-AI image analysis.

This does NOT replace a real trained model — it's a simple greenness-ratio
heuristic so the Camera / AI Hub tab has something real to compute on an
uploaded photo before a proper model is trained and flashed to the
ESP32-S3 hub. Swap `analyze_image()` for a call to the real model's output
(or to whatever the hub reports over LoRa/Wi-Fi) once it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class ImageAnalysis:
    green_ratio: float
    verdict: str
    note: str


def analyze_image(image: Image.Image) -> ImageAnalysis:
    img = image.convert("RGB").resize((160, 160))
    pixels = list(img.getdata())
    green_pixels = sum(1 for r, g, b in pixels if g > r and g > b)
    green_ratio = green_pixels / len(pixels)

    if green_ratio > 0.45:
        verdict = "Healthy canopy"
        note = "High proportion of green pixels — consistent with healthy, actively growing foliage."
    elif green_ratio > 0.2:
        verdict = "Moderate cover"
        note = "Some green cover detected — could be early growth stage, partial coverage, or stress."
    else:
        verdict = "Low green cover"
        note = "Little green detected — check for bare soil, drought stress, disease, or a poorly framed shot."

    return ImageAnalysis(green_ratio=green_ratio, verdict=verdict, note=note)
