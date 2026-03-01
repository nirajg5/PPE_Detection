"""
utils/visualizer.py
===================
Handles all drawing and UI overlay for PPE detection.

Features:
- Bounding boxes with class labels
- Per-person compliance status panels
- PPE worn/missing legend
- FPS counter
- Frame stats bar
- Zone indicator
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time


class Visualizer:
    """Renders all detection results onto frames."""

    # UI Colors (BGR)
    COLOR_BG        = (30, 30, 30)
    COLOR_WHITE     = (255, 255, 255)
    COLOR_GREEN     = (0, 220, 0)
    COLOR_RED       = (0, 0, 220)
    COLOR_YELLOW    = (0, 220, 220)
    COLOR_CYAN      = (220, 220, 0)
    COLOR_ORANGE    = (0, 140, 255)
    COLOR_PURPLE    = (180, 0, 180)
    COLOR_DARK_GREY = (60, 60, 60)

    PPE_COLORS = {
        "hard-hat":       (0, 255, 0),
        "safety-vest":    (0, 255, 255),
        "gloves":         (255, 0, 255),
        "safety-goggles": (0, 200, 255),
        "face-mask":      (100, 200, 100),
        "safety-boots":   (200, 100, 50),
        "harness":        (50, 150, 255),
        "ear-protection": (180, 50, 255),
        "person":         (255, 140, 0),
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX
        self._fps_history = []
        self._last_time = time.time()

    # ------------------------------------------------------------------
    # Main Draw Function
    # ------------------------------------------------------------------

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        compliance_results: List[Dict],
        fps: float = 0.0,
        zone: str = "default",
        paused: bool = False,
    ) -> np.ndarray:
        """
        Draw all annotations on the frame.

        Args:
            frame: Original BGR frame
            detections: Raw detections from PPEDetector
            compliance_results: Compliance results from ComplianceChecker
            fps: Current FPS
            zone: Active compliance zone name
            paused: Whether detection is paused

        Returns:
            Annotated frame
        """
        out = frame.copy()

        # 1. Draw PPE bounding boxes (non-person)
        for det in detections:
            if det["class_name"] != "person":
                self._draw_ppe_box(out, det)

        # 2. Draw person boxes + compliance panels
        for result in compliance_results:
            self._draw_person_compliance(out, result)

        # 3. Top stats bar
        self._draw_stats_bar(out, detections, compliance_results, fps, zone)

        # 4. PPE Legend (bottom-left)
        self._draw_legend(out, detections)

        # 5. Keyboard hints (bottom-right)
        self._draw_hints(out, paused)

        # 6. Paused overlay
        if paused:
            self._draw_paused_overlay(out)

        return out

    # ------------------------------------------------------------------
    # Drawing Helpers
    # ------------------------------------------------------------------

    def _draw_ppe_box(self, frame: np.ndarray, det: Dict):
        """Draw a PPE bounding box with label."""
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        conf = det["confidence"]
        color = self.PPE_COLORS.get(cls, (200, 200, 200))

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, self.font, 0.5, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 2), self.font, 0.5, (0, 0, 0), 1)

    def _draw_person_compliance(self, frame: np.ndarray, result: Dict):
        """Draw person bbox with compliance status panel."""
        x1, y1, x2, y2 = result["person_bbox"]
        compliant = result["compliant"]
        track_id = result.get("track_id", -1)
        worn = result["worn_ppe"]
        missing = result["missing_ppe"]
        score = result["compliance_score"]

        # Person box color
        box_color = self.COLOR_GREEN if compliant else self.COLOR_RED
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Corner decorations
        corner_len = 20
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), box_color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), box_color, 3)

        # --- Compliance Panel (above person) ---
        panel_x = x1
        panel_y = max(0, y1 - 5)
        panel_w = max(200, x2 - x1)
        row_h = 22
        n_rows = 2 + len(missing) + (1 if missing else 0)
        panel_h = row_h * n_rows + 10

        # Semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y - panel_h),
                      (panel_x + panel_w, panel_y),
                      self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Person ID + score
        id_text = f"Person #{track_id}"
        score_text = f"Score: {score:.0f}%"
        score_color = self.COLOR_GREEN if score >= 80 else (self.COLOR_YELLOW if score >= 50 else self.COLOR_RED)

        ty = panel_y - panel_h + row_h
        cv2.putText(frame, id_text, (panel_x + 5, ty),
                    self.font_bold, 0.55, self.COLOR_WHITE, 1)
        cv2.putText(frame, score_text, (panel_x + panel_w - 100, ty),
                    self.font, 0.5, score_color, 1)

        # Score bar
        ty += 6
        bar_bg_end = panel_x + panel_w - 5
        cv2.rectangle(frame, (panel_x + 5, ty), (bar_bg_end, ty + 5), self.COLOR_DARK_GREY, -1)
        bar_fill = int((panel_x + 5) + (bar_bg_end - panel_x - 5) * (score / 100))
        cv2.rectangle(frame, (panel_x + 5, ty), (bar_fill, ty + 5), score_color, -1)

        # Worn PPE list
        ty += row_h
        worn_str = "✓ " + ", ".join(sorted(worn)) if worn else "✓ None"
        cv2.putText(frame, worn_str[:55], (panel_x + 5, ty),
                    self.font, 0.42, self.COLOR_GREEN, 1)

        # Missing PPE list
        if missing:
            ty += row_h - 4
            cv2.putText(frame, "MISSING:", (panel_x + 5, ty),
                        self.font_bold, 0.45, self.COLOR_RED, 1)
            for item in sorted(missing):
                ty += row_h - 4
                cv2.putText(frame, f"  ✗ {item}", (panel_x + 5, ty),
                            self.font, 0.42, self.COLOR_RED, 1)

    def _draw_stats_bar(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        compliance_results: List[Dict],
        fps: float,
        zone: str,
    ):
        """Draw the top statistics bar."""
        h, w = frame.shape[:2]
        bar_h = 40

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        n_persons = len(compliance_results)
        n_compliant = sum(1 for r in compliance_results if r["compliant"])
        n_violation = n_persons - n_compliant
        n_ppe = len([d for d in detections if d["class_name"] != "person"])

        # Title
        cv2.putText(frame, "PPE DETECTION SYSTEM", (10, 28),
                    self.font_bold, 0.7, self.COLOR_CYAN, 1)

        # Stats
        stats = [
            (f"FPS: {fps:.1f}", self.COLOR_WHITE, 260),
            (f"Persons: {n_persons}", self.COLOR_WHITE, 370),
            (f"Compliant: {n_compliant}", self.COLOR_GREEN, 480),
            (f"Violations: {n_violation}", self.COLOR_RED if n_violation > 0 else self.COLOR_WHITE, 600),
            (f"PPE Items: {n_ppe}", self.COLOR_CYAN, 730),
            (f"Zone: {zone.upper()}", self.COLOR_YELLOW, 870),
        ]

        for text, color, x in stats:
            if x < w - 50:
                cv2.putText(frame, text, (x, 27), self.font, 0.55, color, 1)

        # Bottom border line
        cv2.line(frame, (0, bar_h), (w, bar_h), self.COLOR_CYAN, 1)

    def _draw_legend(self, frame: np.ndarray, detections: List[Dict]):
        """Draw PPE class legend in bottom-left."""
        h, w = frame.shape[:2]

        # Get detected classes (excluding person)
        detected_classes = set(
            d["class_name"] for d in detections if d["class_name"] != "person"
        )

        all_ppe = list(self.PPE_COLORS.keys())
        all_ppe = [c for c in all_ppe if c != "person"]

        legend_x = 10
        legend_y_start = h - len(all_ppe) * 20 - 10
        row_h = 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (legend_x - 5, legend_y_start - 5),
                      (legend_x + 155, h - 5),
                      self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, cls_name in enumerate(all_ppe):
            y = legend_y_start + i * row_h
            color = self.PPE_COLORS[cls_name]

            # Filled square indicator
            cv2.rectangle(frame, (legend_x, y - 12), (legend_x + 14, y), color, -1)

            # Text - highlighted if detected
            text_color = self.COLOR_WHITE if cls_name in detected_classes else (120, 120, 120)
            prefix = "●" if cls_name in detected_classes else "○"
            cv2.putText(frame, f"{prefix} {cls_name}", (legend_x + 18, y - 1),
                        self.font, 0.42, text_color, 1)

    def _draw_hints(self, frame: np.ndarray, paused: bool):
        """Draw keyboard shortcut hints."""
        h, w = frame.shape[:2]
        hints = [
            "Q: Quit",
            "P: Pause/Resume",
            "S: Screenshot",
            "Z: Switch Zone",
            "+/-: Confidence",
        ]
        x = w - 140
        y = h - len(hints) * 20 - 10
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (x, y + i * 20),
                        self.font, 0.38, (160, 160, 160), 1)

    def _draw_paused_overlay(self, frame: np.ndarray):
        """Draw PAUSED text overlay."""
        h, w = frame.shape[:2]
        text = "⏸ PAUSED"
        (tw, th), _ = cv2.getTextSize(text, self.font_bold, 2, 3)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.putText(frame, text, (cx + 2, cy + 2), self.font_bold, 2, (0, 0, 0), 3)
        cv2.putText(frame, text, (cx, cy), self.font_bold, 2, self.COLOR_YELLOW, 3)

    def compute_fps(self) -> float:
        """Compute rolling average FPS."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self._fps_history.append(1.0 / dt if dt > 0 else 0)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return float(np.mean(self._fps_history))
