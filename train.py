"""
train.py
========
Fine-tune YOLOv9 on a custom PPE dataset.

Usage:
    python train.py --data data/ppe_dataset.yaml --epochs 100

Dataset format (YOLO format):
    data/
      images/
        train/  *.jpg
        val/    *.jpg
      labels/
        train/  *.txt  (YOLO format: class cx cy w h)
        val/    *.txt

Recommended datasets:
    - Roboflow PPE Dataset: https://universe.roboflow.com/roboflow-100/ppe-raw-images
    - Hard Hat Workers: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv9 for PPE Detection")
    p.add_argument("--data",    type=str, default="data/ppe_dataset.yaml", help="Dataset YAML")
    p.add_argument("--model",   type=str, default="yolov9c.pt",            help="Pretrained weights")
    p.add_argument("--epochs",  type=int, default=100,                     help="Training epochs")
    p.add_argument("--batch",   type=int, default=16,                      help="Batch size")
    p.add_argument("--imgsz",   type=int, default=640,                     help="Image size")
    p.add_argument("--lr",      type=float, default=0.01,                  help="Initial learning rate")
    p.add_argument("--workers", type=int, default=8,                       help="DataLoader workers")
    p.add_argument("--device",  type=str, default="auto",                  help="Device: auto/cpu/cuda")
    p.add_argument("--project", type=str, default="runs/train",            help="Save results to project")
    p.add_argument("--name",    type=str, default="ppe_yolov9",            help="Experiment name")
    p.add_argument("--resume",  action="store_true",                       help="Resume training")
    return p.parse_args()


def create_dataset_yaml(output_path: str = "data/ppe_dataset.yaml"):
    """Generate a template dataset YAML if it doesn't exist."""
    import yaml

    dataset_cfg = {
        "path": "data",
        "train": "images/train",
        "val":   "images/val",
        "nc":    9,
        "names": [
            "person",
            "hard-hat",
            "safety-vest",
            "gloves",
            "safety-goggles",
            "face-mask",
            "safety-boots",
            "harness",
            "ear-protection",
        ]
    }

    path = Path(output_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataset_cfg, f, default_flow_style=False)
        print(f"📄 Created template dataset YAML: {output_path}")
        print("   ⚠️  Edit this file to point to your actual dataset images.")
    else:
        print(f"✅ Dataset YAML found: {output_path}")

    return str(path)


def main():
    args = parse_args()

    print("\n" + "="*55)
    print("  YOLOv9 PPE Detection - Training")
    print("="*55)

    # Create dataset YAML if missing
    dataset_yaml = create_dataset_yaml(args.data)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Resolve device
    device = args.device
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n🔧 Config:")
    print(f"   Model:   {args.model}")
    print(f"   Data:    {dataset_yaml}")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   ImgSz:   {args.imgsz}")
    print(f"   Device:  {device}")
    print(f"   Output:  {args.project}/{args.name}")

    model = YOLO(args.model)

    print(f"\n🚀 Starting training...\n")

    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        patience=50,
        save_period=10,
        resume=args.resume,
        plots=True,
        verbose=True,
    )

    print(f"\n✅ Training complete!")
    print(f"   Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"\nRun inference with:")
    print(f"   python main.py --model {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
