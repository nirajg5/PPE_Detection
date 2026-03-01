"""
utils/detector.py
=================
Core YOLOv9 detection engine for PPE detection.
Handles model loading, inference, result parsing,
frame skipping, and result caching for low-latency operation.

Performance optimizations applied:
  - Frame skip counter   → run inference every N frames, return cache in between
  - Input resize         → shrink frame before inference (faster GPU pass)
  - half_precision FP16  → 2× faster on CUDA GPU
  - Bbox scaling         → results correctly mapped back to original resolution
  - Warmup uses actual img_size (not hardcoded 640)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
from typing import Optional, Tuple, List, Dict


class PPEDetector:
    """
    YOLOv9-based PPE Detection Engine with latency optimizations.

    Detects all PPE classes: person, hard-hat, safety-vest,
    gloves, safety-goggles, face-mask, safety-boots, harness, ear-protection.
    """

    def __init__(
        self,
        model_path: str = "models/yolov9c.pt",
        config_path: str = "config/ppe_config.yaml",
        model_config_path: str = "config/model_config.yaml",
    ):
        """
        Initialize the PPE Detector.

        Args:
            model_path:        Path to YOLOv9 .pt weights file
            config_path:       Path to PPE config YAML
            model_config_path: Path to model config YAML
        """
        # --- Load Configs ---
        self.ppe_cfg   = self._load_yaml(config_path)
        self.model_cfg = self._load_yaml(model_config_path)

        infer = self.model_cfg.get("inference", {})
        self.conf_thresh    = infer.get("confidence_threshold", 0.50)
        self.iou_thresh     = infer.get("iou_threshold",        0.50)
        self.max_det        = infer.get("max_detections",        30)
        self.img_size       = infer.get("input_size",            416)
        self.half_precision = infer.get("half_precision",        False)

        # --- Performance: frame skipping ---
        perf = self.model_cfg.get("performance", {})
        self.skip_frames = perf.get("skip_frames", 2)   # 0 = no skip

        # Internal frame counter + result cache
        self._frame_counter = 0
        self._cached_detections: List[Dict] = []

        # --- Device Setup ---
        device_cfg  = infer.get("device", "auto")
        self.device = self._resolve_device(device_cfg)

        # FP16 only valid on CUDA
        if self.half_precision and self.device != "cuda":
            print("⚠️  half_precision requires CUDA — disabling.")
            self.half_precision = False

        print(f"🔧 Device      : {self.device}")
        print(f"🔧 Input size  : {self.img_size}px")
        print(f"🔧 Skip frames : {self.skip_frames}  (inference every {self.skip_frames + 1} frames)")
        print(f"🔧 FP16        : {self.half_precision}")
        print(f"🔧 Max dets    : {self.max_det}")

        # --- Load Model ---
        self.model = self._load_model(model_path)

        # --- Class Mappings ---
        self.class_names   = self.ppe_cfg.get("classes", {})
        self.class_colors  = self.ppe_cfg.get("class_colors", {})
        self.model_classes = self.model.names   # {0: 'person', 1: 'hard-hat', ...}

        print(f"✅ PPEDetector ready | Classes: {list(self.model_classes.values())}\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv9 inference on a single frame (no tracking).
        Applies frame-skip caching to reduce CPU/GPU load.

        Args:
            frame: BGR image from OpenCV (any resolution)

        Returns:
            List of detection dicts with keys:
            class_id, class_name, confidence, bbox [x1,y1,x2,y2],
            bbox_norm, center (cx,cy), track_id, width, height
        """
        # --- Frame skip: return cached result on skipped frames ---
        self._frame_counter += 1
        if self.skip_frames > 0 and self._frame_counter % (self.skip_frames + 1) != 0:
            return self._cached_detections

        # --- Resize frame for faster inference ---
        resized = self._resize_for_inference(frame)

        results = self.model.predict(
            source=resized,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=self.max_det,
            imgsz=self.img_size,
            device=self.device,
            half=self.half_precision,
            verbose=False,
        )

        self._cached_detections = self._parse_results(results, frame, tracked=False)
        return self._cached_detections

    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv9 inference WITH ByteTrack tracking.
        Applies frame-skip caching to reduce CPU/GPU load.

        Args:
            frame: BGR image from OpenCV (any resolution)

        Returns:
            Same as detect() but track_id is always an int (not None).
        """
        # --- Frame skip: return cached result on skipped frames ---
        self._frame_counter += 1
        if self.skip_frames > 0 and self._frame_counter % (self.skip_frames + 1) != 0:
            return self._cached_detections

        tracker_cfg  = self.model_cfg.get("tracker", {})
        tracker_type = tracker_cfg.get("type", "bytetrack")

        # --- Resize frame for faster inference ---
        resized = self._resize_for_inference(frame)

        results = self.model.track(
            source=resized,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=self.max_det,
            imgsz=self.img_size,
            device=self.device,
            half=self.half_precision,
            tracker=f"{tracker_type}.yaml",
            persist=True,
            verbose=False,
        )

        self._cached_detections = self._parse_results(results, frame, tracked=True)
        return self._cached_detections

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Return BGR color tuple for a PPE class name."""
        colors = self.class_colors.get(class_name, [200, 200, 200])
        return tuple(colors)

    def warmup(self):
        """
        Run a dummy inference to JIT-compile the model.
        Uses the configured img_size so warmup matches production input.
        """
        sz = self.img_size
        dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, half=self.half_precision)
        print("🔥 Model warmed up\n")

    def reset_cache(self):
        """Clear cached detections and reset frame counter (e.g. after pause)."""
        self._cached_detections = []
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _resize_for_inference(self, frame: np.ndarray) -> np.ndarray:
        """
        Downscale the camera frame before passing to YOLOv9.

        Scales the larger dimension to img_size while preserving aspect ratio.
        This is the single biggest latency win on high-res cameras (1280×720).

        Returns the resized frame (or original if already small enough).
        """
        h, w = frame.shape[:2]
        target = self.img_size

        # Already fits within target — no resize needed
        if w <= target and h <= target:
            return frame

        scale   = target / max(h, w)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _parse_results(
        self,
        results,
        original_frame: np.ndarray,
        tracked: bool = False,
    ) -> List[Dict]:
        """
        Convert ultralytics Results into clean detection dicts.

        Bounding boxes are scaled from the resized-inference space back
        to the ORIGINAL camera frame dimensions so downstream code
        (compliance checker, visualizer) always sees correct pixel coords.

        Args:
            results:        ultralytics Results list
            original_frame: full-resolution camera frame (for shape reference)
            tracked:        whether track IDs should be populated

        Returns:
            List[Dict] — one dict per detected object
        """
        detections: List[Dict] = []
        oh, ow = original_frame.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # orig_shape = the shape of the image the model actually received
            if hasattr(result, "orig_shape") and result.orig_shape:
                rh, rw = result.orig_shape[:2]
            else:
                rh, rw = oh, ow

            # Scale factors: inference-space → original-frame-space
            sx = ow / rw if rw > 0 else 1.0
            sy = oh / rh if rh > 0 else 1.0

            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)

                    # Map back to original resolution
                    x1 = int(x1 * sx);  y1 = int(y1 * sy)
                    x2 = int(x2 * sx);  y2 = int(y2 * sy)

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, ow - 1))
                    y1 = max(0, min(y1, oh - 1))
                    x2 = max(0, min(x2, ow - 1))
                    y2 = max(0, min(y2, oh - 1))

                    conf     = float(box.conf[0].cpu())
                    cls_id   = int(box.cls[0].cpu())
                    cls_name = self.model_classes.get(cls_id, f"class_{cls_id}")

                    # Track ID
                    track_id = None
                    if tracked:
                        track_id = int(box.id[0].cpu()) if box.id is not None else -1

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    detections.append({
                        "class_id":   cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox":       [x1, y1, x2, y2],
                        "bbox_norm":  [x1 / ow, y1 / oh, x2 / ow, y2 / oh],
                        "center":     (cx, cy),
                        "track_id":   track_id,
                        "width":      x2 - x1,
                        "height":     y2 - y1,
                    })

                except Exception:
                    # Skip any malformed box silently
                    continue

        return detections

    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLOv9 .pt weights, auto-downloading if the file is missing."""
        path = Path(model_path)
        if not path.exists():
            print(f"⚠️  Model not found at {path}. Attempting auto-download...")
            try:
                from models.download_model import download_model
                variant = path.stem   # e.g. "yolov9c" from "models/yolov9c.pt"
                download_model(variant)
            except Exception as e:
                raise FileNotFoundError(
                    f"Model not found and download failed: {e}\n"
                    "Please manually place the .pt file in the models/ folder."
                )

        model = YOLO(str(path))
        return model

    @staticmethod
    def _resolve_device(device_cfg: str) -> str:
        """Resolve 'auto' to the best available torch device."""
        if device_cfg == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_cfg

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """Load a YAML config file and return as dict."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
