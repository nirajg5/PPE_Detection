"""
main.py
=======
🚀 PPE Detection System - Main Entry Point

Real-time PPE detection using YOLOv9 + laptop webcam.
Optimized for low latency: threaded inference, display resize,
frame-skip caching, and minimal blocking in the main loop.

Usage:
    python main.py
    python main.py --camera 0
    python main.py --confidence 0.5
    python main.py --save-video
    python main.py --zone construction_zone
    python main.py --no-alerts
    python main.py --no-track          (faster, no person IDs)

Keyboard Controls (while running):
    Q / ESC  - Quit
    P        - Pause / Resume
    S        - Save Screenshot
    Z        - Cycle through zones
    +        - Increase confidence threshold
    -        - Decrease confidence threshold
"""

import argparse
import sys
import os
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from utils.detector    import PPEDetector
from utils.compliance  import ComplianceChecker
from utils.visualizer  import Visualizer
from utils.logger      import ViolationLogger
from alerts.alert_manager import AlertManager

# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------
ZONES = [
    "default",
    "construction_zone",
    "chemical_zone",
    "height_zone",
    "noise_zone",
]

# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv9 PPE Detection System")
    p.add_argument("--camera",      type=int,   default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--model",       type=str,   default="models/yolov9c.pt",
                   help="Path to YOLOv9 weights")
    p.add_argument("--confidence",  type=float, default=0.50,
                   help="Detection confidence threshold (0–1)")
    p.add_argument("--iou",         type=float, default=0.50,
                   help="NMS IoU threshold (0–1)")
    p.add_argument("--width",       type=int,   default=640,
                   help="Camera capture width  (default: 640 for low latency)")
    p.add_argument("--height",      type=int,   default=480,
                   help="Camera capture height (default: 480 for low latency)")
    p.add_argument("--display-w",   type=int,   default=960,
                   help="Display window width  (independent of capture size)")
    p.add_argument("--display-h",   type=int,   default=540,
                   help="Display window height (independent of capture size)")
    p.add_argument("--zone",        type=str,   default="default",
                   choices=ZONES, help="Compliance zone")
    p.add_argument("--save-video",  action="store_true",
                   help="Save output video to logs/")
    p.add_argument("--no-track",    action="store_true",
                   help="Disable ByteTrack (faster, no person IDs)")
    p.add_argument("--no-alerts",   action="store_true",
                   help="Disable sound alerts")
    p.add_argument("--no-log",      action="store_true",
                   help="Disable violation logging")
    p.add_argument("--fullscreen",  action="store_true",
                   help="Open in fullscreen mode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open and configure the webcam for low-latency capture."""
    print(f"📷 Opening camera index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows

    if not cap.isOpened():
        # Fallback: try without backend flag
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"❌ Cannot open camera {index}. "
            "Try --camera 1 or check if the camera is connected."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # 1-frame buffer → minimal latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera opened: {actual_w}×{actual_h}")
    return cap


# ---------------------------------------------------------------------------
# Video Writer
# ---------------------------------------------------------------------------

def create_video_writer(log_dir: str, w: int, h: int):
    """Create an MP4 video writer for saving output."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(log_dir) / f"ppe_detection_{ts}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
    print(f"🎬 Saving video to: {out_path}")
    return writer, out_path


# ---------------------------------------------------------------------------
# Threaded Inference Worker
# ---------------------------------------------------------------------------

class InferenceThread(threading.Thread):
    """
    Runs detection in a background thread so the main loop
    never blocks waiting for the GPU/CPU — eliminating the
    primary source of UI hang / high latency.

    The main loop reads results via .get_results() which
    always returns instantly (last completed detection).
    """

    def __init__(self, detector: PPEDetector, compliance: ComplianceChecker,
                 logger, alert_mgr):
        super().__init__(daemon=True)
        self.detector   = detector
        self.compliance = compliance
        self.logger     = logger
        self.alert_mgr  = alert_mgr

        self._lock          = threading.Lock()
        self._frame_queue   = deque(maxlen=1)   # only keep latest frame
        self._detections    = []
        self._compliance    = []
        self._frame_id      = 0
        self._running       = True
        self._use_tracking  = True

    def submit_frame(self, frame: np.ndarray, frame_id: int):
        """Push a new frame for processing (drops old unprocessed frames)."""
        self._frame_queue.append((frame.copy(), frame_id))

    def get_results(self):
        """Return the latest (detections, compliance_results) — non-blocking."""
        with self._lock:
            return self._detections, self._compliance

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            if not self._frame_queue:
                time.sleep(0.005)   # yield CPU briefly when no frames waiting
                continue

            frame, frame_id = self._frame_queue.pop()

            try:
                if self._use_tracking:
                    dets = self.detector.detect_and_track(frame)
                else:
                    dets = self.detector.detect(frame)

                comp = self.compliance.check_frame(dets)

                with self._lock:
                    self._detections = dets
                    self._compliance = comp

                # Logging + alerts happen in inference thread (off main loop)
                if self.logger:
                    self.logger.log(comp, frame=frame, frame_id=frame_id)
                if self.alert_mgr:
                    self.alert_mgr.process_violations(comp)

            except Exception as e:
                print(f"⚠️  Inference error (frame {frame_id}): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "=" * 62)
    print("  🦺  PPE DETECTION SYSTEM  —  YOLOv9 + Real-Time Camera")
    print("=" * 62)

    # ----------------------------------------------------------------
    # Initialize components
    # ----------------------------------------------------------------
    print("\n🔧 Initializing components...")

    detector = PPEDetector(
        model_path=args.model,
        config_path="config/ppe_config.yaml",
        model_config_path="config/model_config.yaml",
    )
    # CLI overrides
    detector.conf_thresh = args.confidence
    detector.iou_thresh  = args.iou

    compliance_checker = ComplianceChecker(config_path="config/ppe_config.yaml")
    compliance_checker.set_zone(args.zone)

    visualizer = Visualizer()

    logger = None
    if not args.no_log:
        logger = ViolationLogger(
            log_dir="logs",
            screenshot_dir="logs/screenshots",
            log_format="csv",
            screenshot_on_violation=True,
        )

    alert_mgr = None
    if not args.no_alerts:
        alert_mgr = AlertManager(sound_enabled=True, alert_cooldown_seconds=5.0)

    # ----------------------------------------------------------------
    # Warmup
    # ----------------------------------------------------------------
    print("🔥 Warming up model...")
    detector.warmup()

    # ----------------------------------------------------------------
    # Camera
    # ----------------------------------------------------------------
    cap      = open_camera(args.camera, args.width, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Display resolution (can be different from capture resolution)
    disp_w = args.display_w
    disp_h = args.display_h

    # ----------------------------------------------------------------
    # Video writer (uses display resolution)
    # ----------------------------------------------------------------
    video_writer = None
    if args.save_video:
        video_writer, _ = create_video_writer("logs", disp_w, disp_h)

    # ----------------------------------------------------------------
    # Window
    # ----------------------------------------------------------------
    win_name = "PPE Detection — YOLOv9 | Q=Quit  P=Pause  S=Screenshot  Z=Zone"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win_name, disp_w, disp_h)

    # ----------------------------------------------------------------
    # Start background inference thread
    # ----------------------------------------------------------------
    infer_thread = InferenceThread(
        detector=detector,
        compliance=compliance_checker,
        logger=logger,
        alert_mgr=alert_mgr,
    )
    infer_thread._use_tracking = not args.no_track
    infer_thread.start()

    # ----------------------------------------------------------------
    # State
    # ----------------------------------------------------------------
    paused      = False
    frame_id    = 0
    zone_idx    = ZONES.index(args.zone)
    active_zone = args.zone

    print("\n✅ All systems ready! Starting detection...\n")
    print("  Q=Quit | P=Pause | S=Screenshot | Z=Zone | +/-=Confidence\n")

    # ----------------------------------------------------------------
    # Main loop  (UI thread — never blocks on inference)
    # ----------------------------------------------------------------
    try:
        while True:
            # -- Read frame (non-blocking with buffer=1) --
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Camera read failed. Retrying...")
                time.sleep(0.03)
                continue

            frame_id += 1

            # -- Submit frame to background thread --
            if not paused:
                infer_thread.submit_frame(frame, frame_id)

            # -- Get latest results (always instant) --
            detections, compliance_results = infer_thread.get_results()

            # -- FPS --
            fps = visualizer.compute_fps()

            # -- Draw annotations --
            annotated = visualizer.draw_frame(
                frame=frame,
                detections=detections,
                compliance_results=compliance_results,
                fps=fps,
                zone=active_zone,
                paused=paused,
            )

            # -- Resize for display (reduces rendering overhead) --
            display_frame = cv2.resize(annotated, (disp_w, disp_h),
                                       interpolation=cv2.INTER_LINEAR)

            # -- Save to video --
            if video_writer:
                video_writer.write(display_frame)

            # -- Show --
            cv2.imshow(win_name, display_frame)

            # ----------------------------------------------------------
            # Key handling  (waitKey(1) = 1 ms — non-blocking)
            # ----------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):          # Q / ESC → Quit
                print("\n👋 Quit requested.")
                break

            elif key == ord("p"):              # P → Pause / Resume
                paused = not paused
                if paused:
                    detector.reset_cache()     # clear stale cache on resume
                print(f"⏸  {'Paused' if paused else 'Resumed'}")

            elif key == ord("s"):              # S → Screenshot
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("logs/screenshots", exist_ok=True)
                path = f"logs/screenshots/screenshot_{ts_str}.jpg"
                cv2.imwrite(path, display_frame)
                print(f"📸 Screenshot saved: {path}")

            elif key == ord("z"):              # Z → Cycle zone
                zone_idx    = (zone_idx + 1) % len(ZONES)
                active_zone = ZONES[zone_idx]
                compliance_checker.set_zone(active_zone)

            elif key in (ord("+"), ord("=")):  # + → Raise confidence
                detector.conf_thresh = min(0.95, detector.conf_thresh + 0.05)
                print(f"🎯 Confidence: {detector.conf_thresh:.2f}")

            elif key == ord("-"):              # - → Lower confidence
                detector.conf_thresh = max(0.10, detector.conf_thresh - 0.05)
                print(f"🎯 Confidence: {detector.conf_thresh:.2f}")

    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user.")

    finally:
        # ---------------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------------
        print("\n🧹 Cleaning up...")
        infer_thread.stop()
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # Session summary
        if logger:
            summary = logger.get_session_summary()
            print("\n📊 Session Summary:")
            print(f"   Violations logged   : {summary['total_violations_logged']}")
            print(f"   Unique persons      : {summary['unique_persons_violated']}")
            if summary["most_common_missing"]:
                print(f"   Most missed PPE     : {summary['most_common_missing']}")
            print(f"   Log file            : {summary['log_file']}")

        print("\n✅ PPE Detection System stopped.\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
