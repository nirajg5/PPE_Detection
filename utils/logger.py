"""
utils/logger.py
===============
Logs PPE violations to CSV and JSON files.
Optionally saves screenshot frames on violation.
"""

import csv
import json
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class ViolationLogger:
    """
    Logs PPE violations with timestamps, track IDs, and missing PPE details.
    Supports CSV and JSON formats, plus optional screenshot capture.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        screenshot_dir: str = "logs/screenshots",
        log_format: str = "csv",
        screenshot_on_violation: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.screenshot_dir = Path(screenshot_dir)
        self.log_format = log_format
        self.screenshot_enabled = screenshot_on_violation

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Session log file
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"violations_{session_ts}.csv"
        self.json_path = self.log_dir / f"violations_{session_ts}.json"

        self._json_entries = []
        self._csv_initialized = False

        # Track last violation time per person (to avoid log spam)
        self._last_log_time: Dict[int, float] = {}
        self._log_cooldown = 5.0  # seconds

        print(f"📋 ViolationLogger initialized | Log: {self.csv_path}")

    def log(
        self,
        compliance_results: List[Dict],
        frame: Optional[np.ndarray] = None,
        frame_id: int = 0,
    ):
        """
        Log violations from compliance results.

        Args:
            compliance_results: Output of ComplianceChecker.check_frame()
            frame: Current video frame (for screenshots)
            frame_id: Current frame number
        """
        timestamp = datetime.now().isoformat()
        now = time.time()

        for result in compliance_results:
            if result["compliant"]:
                continue

            track_id = result.get("track_id", -1)

            # Cooldown check
            if track_id in self._last_log_time:
                if now - self._last_log_time[track_id] < self._log_cooldown:
                    continue

            self._last_log_time[track_id] = now

            entry = {
                "timestamp": timestamp,
                "frame_id": frame_id,
                "track_id": track_id,
                "zone": result.get("zone", "default"),
                "worn_ppe": list(result["worn_ppe"]),
                "missing_ppe": list(result["missing_ppe"]),
                "compliance_score": result["compliance_score"],
            }

            # Write CSV
            if self.log_format in ("csv", "both"):
                self._write_csv(entry)

            # Write JSON
            if self.log_format in ("json", "both"):
                self._json_entries.append(entry)
                self._write_json()

            # Screenshot
            if self.screenshot_enabled and frame is not None:
                self._save_screenshot(frame, track_id, timestamp)

    def log_manual_screenshot(self, frame: np.ndarray, label: str = "manual"):
        """Save a manually triggered screenshot."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.screenshot_dir / f"screenshot_{label}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"📸 Screenshot saved: {path}")
        return str(path)

    def get_session_summary(self) -> Dict:
        """Return summary stats for this session."""
        return {
            "total_violations_logged": len(self._json_entries),
            "unique_persons_violated": len(set(e["track_id"] for e in self._json_entries)),
            "most_common_missing": self._most_common_missing(),
            "log_file": str(self.csv_path),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _write_csv(self, entry: Dict):
        """Append one row to CSV file."""
        fieldnames = ["timestamp", "frame_id", "track_id", "zone",
                      "worn_ppe", "missing_ppe", "compliance_score"]

        write_header = not self._csv_initialized
        self._csv_initialized = True

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            row = entry.copy()
            row["worn_ppe"] = "|".join(row["worn_ppe"])
            row["missing_ppe"] = "|".join(row["missing_ppe"])
            writer.writerow(row)

    def _write_json(self):
        """Write all entries to JSON file."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_entries, f, indent=2)

    def _save_screenshot(self, frame: np.ndarray, track_id: int, timestamp: str):
        """Save violation screenshot."""
        ts_clean = timestamp.replace(":", "-").replace(".", "-")
        path = self.screenshot_dir / f"violation_person{track_id}_{ts_clean}.jpg"
        cv2.imwrite(str(path), frame)

    def _most_common_missing(self) -> Dict[str, int]:
        """Count most frequently missing PPE items."""
        counts: Dict[str, int] = {}
        for entry in self._json_entries:
            for item in entry["missing_ppe"]:
                counts[item] = counts.get(item, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
