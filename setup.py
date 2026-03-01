"""
setup.py
========
Auto-setup script for PPE Detection System.
Downloads YOLOv9 model and verifies environment.

Run: python setup.py
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python():
    version = sys.version_info
    if version.major < 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required. Found: {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def install_requirements():
    print("\n📦 Installing requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌ Failed to install some requirements.")
        print("   Try: pip install -r requirements.txt manually")
    else:
        print("✅ Requirements installed")


def download_model():
    print("\n📥 Downloading YOLOv9c model weights...")
    model_path = Path("models/yolov9c.pt")
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return

    try:
        # Try ultralytics auto-download first
        from ultralytics import YOLO
        model = YOLO("yolov9c.pt")
        # Move to models dir
        import shutil
        Path("models").mkdir(exist_ok=True)
        src = Path("yolov9c.pt")
        if src.exists():
            shutil.move(str(src), str(model_path))
            print(f"✅ Model saved to {model_path}")
        else:
            print("✅ Model downloaded by ultralytics (cached)")
    except Exception as e:
        print(f"⚠️  Ultralytics auto-download failed: {e}")
        print("   Trying manual download...")
        try:
            from models.download_model import download_model as dl
            dl("yolov9c")
        except Exception as e2:
            print(f"❌ Manual download failed: {e2}")
            print("   Please manually download yolov9c.pt and place in models/")


def check_camera():
    print("\n📷 Checking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("✅ Camera (index 0) working")
            else:
                print("⚠️  Camera opened but cannot read frames")
        else:
            print("⚠️  No camera at index 0. Try --camera 1 when running main.py")
    except Exception as e:
        print(f"⚠️  Camera check failed: {e}")


def create_dirs():
    """Ensure all necessary directories exist."""
    dirs = [
        "models", "logs", "logs/screenshots",
        "config", "utils", "alerts", "data/images", "data/labels"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")


def main():
    print("="*55)
    print("  PPE Detection System - Setup")
    print("="*55)

    check_python()
    create_dirs()
    install_requirements()
    download_model()
    check_camera()

    print("\n" + "="*55)
    print("✅ Setup complete!")
    print("\nRun the system with:")
    print("   python main.py")
    print("\nOptions:")
    print("   python main.py --camera 1          # Different camera")
    print("   python main.py --confidence 0.5    # Higher confidence")
    print("   python main.py --zone construction_zone")
    print("   python main.py --save-video")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
