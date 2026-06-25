"""
Adaptive Preprocessor — conditional preprocessing pipeline.

Calls MotionGate first. If no motion, skips everything.
Then conditionally runs dehazer, rain filter, and CLAHE based
on per-frame quality metrics instead of blindly applying all three.

Target: <8ms when preprocessing needed, <1ms when skipped.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from .motion_gate import MotionGate, MotionResult
from .dehazer import Dehazer
from .rain_filter import RainFilter
from .enhancer import Enhancer

logger = logging.getLogger(__name__)


class AdaptivePreprocessor:
    """
    Master preprocessor that gates all operations on actual need.

    Pipeline order:
      1. MotionGate  → skip everything if static
      2. Dehazer     → only if foggy (bright + high dark channel)
      3. RainFilter  → only if vertical streaks detected
      4. Enhancer    → only if low contrast
    """

    def __init__(self, config: dict) -> None:
        pcfg = config.get("preprocessing", {})

        self._adaptive: bool = pcfg.get("adaptive", True)
        self._brightness_fog_threshold: float = pcfg.get(
            "brightness_fog_threshold", 160.0
        )
        self._dark_channel_fog_threshold: float = pcfg.get(
            "dark_channel_fog_threshold", 0.08
        )

        self.motion_gate = MotionGate(config)
        self._dehazer = Dehazer(config)
        self._rain_filter = RainFilter(config)
        self._enhancer = Enhancer(config)

        logger.info("AdaptivePreprocessor initialised (adaptive=%s)", self._adaptive)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, MotionResult]:
        """
        Run the adaptive preprocessing pipeline.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from capture.

        Returns
        -------
        tuple[np.ndarray, MotionResult]
            (processed_frame, motion_result)
        """
        # ── 1. Motion gate — always runs ─────────────────────────
        motion_result = self.motion_gate.analyze(frame)

        if not motion_result.has_motion:
            return frame, motion_result

        if not self._adaptive:
            # Non-adaptive mode: run everything unconditionally
            frame = self._dehazer.remove(frame)
            frame = self._rain_filter.remove(frame)
            frame = self._enhancer.enhance(frame)
            return frame, motion_result

        # ── 2. Conditional dehazing ──────────────────────────────
        if self._should_dehaze(frame):
            frame = self._dehazer.remove(frame)

        # ── 3. Conditional rain removal ──────────────────────────
        if self._rain_filter.has_rain(frame):
            frame = self._rain_filter.remove(frame)

        # ── 4. Conditional contrast enhancement ──────────────────
        if self._enhancer.needs_enhancement(frame):
            frame = self._enhancer.enhance(frame)

        return frame, motion_result

    def _should_dehaze(self, frame: np.ndarray) -> bool:
        """
        Check if the frame appears foggy based on brightness and
        dark channel analysis.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness <= self._brightness_fog_threshold:
            return False

        # Quick dark channel check on downscaled image for speed
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        min_channel = np.min(small.astype(np.float64) / 255.0, axis=2)
        dark_mean = float(np.mean(min_channel))

        return dark_mean > self._dark_channel_fog_threshold

    def reset(self) -> None:
        """Reset all sub-components."""
        self.motion_gate.reset()
        self._dehazer.reset()
        self._rain_filter.reset()
        self._enhancer.reset()
        logger.info("AdaptivePreprocessor reset")

    def __repr__(self) -> str:
        return f"AdaptivePreprocessor(adaptive={self._adaptive})"