"""
evaluate.py
===========
Evaluate PPE detection model performance.

Reports:
- mAP@0.5, mAP@0.5:0.95
- Per-class Precision, Recall, F1
- FPS benchmark
- Confusion matrix

Usage:
    python evaluate.py
    python evaluate.py --model models/yolov9c.pt --data data/ppe_dataset.yaml
    python evaluate.py --benchmark   # FPS only, no dataset needed
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPE Detection Model")
    p.add_argument("--model",     type=str, default="models/yolov9c.pt")
    p.add_argument("--data",      type=str, default="data/ppe_dataset.yaml")
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--conf",      type=float, default=0.45)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--benchmark", action="store_true", help="Run FPS benchmark only")
    p.add_argument("--camera",    type=int, default=0, help="Camera for benchmark")
    return p.parse_args()


def fps_benchmark(model_path: str, camera_idx: int = 0, duration: int = 10):
    """Measure real-time FPS on webcam."""
    print(f"\n🏎️  FPS Benchmark ({duration}s on camera {camera_idx})...")

    from ultralytics import YOLO
    model = YOLO(model_path)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("❌ Cannot open camera for benchmark")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_times = []
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        model.predict(frame, verbose=False, conf=0.45)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

    cap.release()

    if frame_times:
        avg_ms = np.mean(frame_times) * 1000
        fps = 1.0 / np.mean(frame_times)
        min_fps = 1.0 / np.max(frame_times)
        max_fps = 1.0 / np.min(frame_times)
        print(f"\n📊 Benchmark Results:")
        print(f"   Frames tested : {len(frame_times)}")
        print(f"   Avg latency   : {avg_ms:.1f} ms")
        print(f"   Avg FPS       : {fps:.1f}")
        print(f"   Min FPS       : {min_fps:.1f}")
        print(f"   Max FPS       : {max_fps:.1f}")
        print(f"   Real-time?    : {'✅ Yes' if fps >= 15 else '⚠️ Below 15 FPS'}")


def evaluate_dataset(model_path: str, data_yaml: str, args):
    """Run full evaluation on a validation dataset."""
    from ultralytics import YOLO

    print(f"\n📊 Evaluating model on dataset: {data_yaml}")
    model = YOLO(model_path)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=True,
    )

    print("\n" + "="*55)
    print("📋 EVALUATION RESULTS")
    print("="*55)

    # Overall metrics
    print(f"\n🎯 Overall:")
    print(f"   mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"   Precision     : {metrics.box.mp:.4f}")
    print(f"   Recall        : {metrics.box.mr:.4f}")

    # Per-class
    print(f"\n📦 Per-Class Metrics:")
    class_names = model.names
    header = f"{'Class':<20} {'AP50':>8} {'Precision':>10} {'Recall':>8}"
    print(f"   {header}")
    print(f"   {'-'*50}")

    try:
        for i, (ap, p, r) in enumerate(zip(
            metrics.box.ap50,
            metrics.box.p,
            metrics.box.r
        )):
            name = class_names.get(i, f"class_{i}")
            print(f"   {name:<20} {ap:>8.3f} {p:>10.3f} {r:>8.3f}")
    except Exception:
        print("   (Per-class metrics not available in this ultralytics version)")

    print(f"\n✅ Evaluation complete.")
    return metrics


def main():
    args = parse_args()

    print("\n" + "="*55)
    print("  YOLOv9 PPE Detection - Evaluation")
    print("="*55)

    if args.benchmark:
        fps_benchmark(args.model, args.camera)
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"⚠️  Dataset YAML not found: {data_path}")
            print("   Running FPS benchmark instead...")
            fps_benchmark(args.model, args.camera)
        else:
            evaluate_dataset(args.model, str(data_path), args)
            fps_benchmark(args.model, args.camera, duration=5)


if __name__ == "__main__":
    main()
