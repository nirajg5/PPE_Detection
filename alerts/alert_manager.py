"""
alerts/alert_manager.py
========================
Handles violation alerts:
- Audio beep (winsound on Windows, pygame on others)
- Console alerts with ANSI color
- Cooldown management to prevent spam

NOTE: Uses only built-in or easily installable audio libs.
      No playsound dependency (known Python 3.11 build issue).
"""

import time
import sys
import threading
from typing import List, Dict


class AlertManager:
    """
    Manages PPE violation alerts with cooldown support.

    Alert priority (auto-selected):
      Windows  → winsound.Beep (built-in, no install needed)
      macOS    → subprocess afplay
      Linux    → subprocess aplay / paplay
      Fallback → pygame mixer sine tone
    """

    def __init__(
        self,
        sound_enabled: bool = True,
        alert_cooldown_seconds: float = 5.0,
        sound_file: str = None,
    ):
        self.sound_enabled = sound_enabled
        self.cooldown = alert_cooldown_seconds
        self.sound_file = sound_file

        # Last alert time per person {track_id: timestamp}
        self._last_alert: Dict[int, float] = {}

        # Detect best audio backend
        self._audio_backend = None
        if self.sound_enabled:
            self._audio_backend = self._detect_audio_backend()
            print(f"🔊 Audio backend: {self._audio_backend}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_violations(self, compliance_results: List[Dict]):
        """
        Check results and trigger alerts for violations.

        Args:
            compliance_results: From ComplianceChecker.check_frame()
        """
        for result in compliance_results:
            if result["compliant"]:
                continue

            track_id = result.get("track_id", -1)
            now = time.time()

            # Cooldown check
            last = self._last_alert.get(track_id, 0)
            if now - last < self.cooldown:
                continue

            self._last_alert[track_id] = now
            self._trigger_alert(track_id, result)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _trigger_alert(self, track_id: int, result: Dict):
        """Fire console + sound alerts."""
        missing = sorted(result["missing_ppe"])
        zone = result.get("zone", "default")

        self._console_alert(track_id, missing, zone)

        if self.sound_enabled and self._audio_backend:
            threading.Thread(target=self._play_sound, daemon=True).start()

    def _console_alert(self, track_id: int, missing: List[str], zone: str):
        """Print ANSI-colored console alert."""
        ts = time.strftime("%H:%M:%S")
        missing_str = ", ".join(missing)
        print(
            f"\033[91m[{ts}] ⚠️  VIOLATION | "
            f"Person #{track_id} | Zone: {zone} | "
            f"Missing: {missing_str}\033[0m"
        )

    def _play_sound(self):
        """Play alert via detected backend."""
        try:
            if self._audio_backend == "winsound":
                import winsound
                # Two short beeps
                winsound.Beep(1200, 200)
                time.sleep(0.05)
                winsound.Beep(1000, 200)

            elif self._audio_backend == "afplay":
                import subprocess
                # macOS built-in: play system alert sound
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Ping.aiff"],
                    capture_output=True, timeout=2
                )

            elif self._audio_backend == "aplay":
                import subprocess
                subprocess.run(
                    ["aplay", "-q", "/usr/share/sounds/alsa/Front_Left.wav"],
                    capture_output=True, timeout=2
                )

            elif self._audio_backend == "pygame":
                self._pygame_beep()

            else:
                # Last resort: terminal bell
                print("\a", end="", flush=True)

        except Exception:
            print("\a", end="", flush=True)

    def _pygame_beep(self):
        """Generate a beep tone using pygame (cross-platform fallback)."""
        try:
            import pygame
            import numpy as np

            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            sample_rate = 44100
            duration = 0.25   # seconds
            freq = 1100        # Hz

            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            time.sleep(duration + 0.05)
        except Exception:
            print("\a", end="", flush=True)

    def _detect_audio_backend(self) -> str:
        """Auto-detect the best available audio backend."""
        # 1. Windows winsound (built-in, always works on Windows)
        if sys.platform == "win32":
            try:
                import winsound
                return "winsound"
            except ImportError:
                pass

        # 2. macOS afplay (built-in)
        if sys.platform == "darwin":
            import shutil
            if shutil.which("afplay"):
                return "afplay"

        # 3. Linux aplay
        if sys.platform.startswith("linux"):
            import shutil
            if shutil.which("aplay"):
                return "aplay"
            if shutil.which("paplay"):
                return "aplay"  # same handler

        # 4. pygame fallback (cross-platform, needs pip install pygame)
        try:
            import pygame
            import numpy
            return "pygame"
        except ImportError:
            pass

        # 5. Nothing available
        print("⚠️  No audio backend found. Alerts will be console-only.")
        return None
