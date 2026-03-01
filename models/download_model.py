"""
models/download_model.py
========================
Auto-downloads YOLOv9 pretrained weights.
Run directly: python models/download_model.py
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

MODELS = {
    "yolov9t": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt",
    "yolov9s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt",
    "yolov9m": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9m.pt",
    "yolov9c": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt",
    "yolov9e": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt",
}

MODEL_DIR = Path(__file__).parent


def download_model(variant: str = "yolov9c", force: bool = False) -> Path:
    """
    Download YOLOv9 model weights.

    Args:
        variant: Model variant (yolov9t/s/m/c/e)
        force: Force re-download even if file exists

    Returns:
        Path to downloaded model
    """
    if variant not in MODELS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(MODELS.keys())}")

    url = MODELS[variant]
    save_path = MODEL_DIR / f"{variant}.pt"

    if save_path.exists() and not force:
        print(f"✅ Model already exists: {save_path}")
        return save_path

    print(f"📥 Downloading {variant} from:\n   {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f, tqdm(
        desc=f"Downloading {variant}.pt",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"✅ Downloaded: {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")
    return save_path


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "yolov9c"
    download_model(variant)
