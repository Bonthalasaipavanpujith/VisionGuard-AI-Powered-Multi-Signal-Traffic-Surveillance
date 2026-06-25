"""
Rain Filter — Temporal weighted averaging for rain streak removal.

Rain streaks are random per frame but the background is consistent.
A weighted average of recent frames cancels streaks while preserving
the scene. Only activated when vertical streak score exceeds threshold.

Reduced to 3-frame buffer with weights [0.6, 0.3, 0.1] — recent
frames matter most to minimise motion blur on moving vehicles.
"""

import logging

import cv2
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class RainFilter:
    """
    Removes rain streaks using a short temporal weighted average.

    Activation is conditional: only runs when the vertical Sobel
    response indicates rain-like vertical streaks in the frame.
    """

    # Fixed weights for 3-frame buffer (most recent first)
    _WEIGHTS = np.array([0.1, 0.3, 0.6], dtype=np.float32)

    def __init__(self, config: dict) -> None:
        pcfg = config.get("preprocessing", {})

        self._num_frames: int = pcfg.get("rain_temporal_frames", 3)
        self._streak_threshold: float = pcfg.get("rain_streak_threshold", 0.3)

        self._frame_buffer: deque = deque(maxlen=self._num_frames)

        # Pre-build vertical Sobel kernel
        self._sobel_kernel_size: int = 3

        logger.info(
            "RainFilter initialised: frames=%d, streak_threshold=%.2f",
            self._num_frames,
            self._streak_threshold,
        )

    def has_rain(self, frame: np.ndarray) -> bool:
        """
        Estimate whether the frame contains rain-like vertical streaks.

        Uses vertical Sobel filter — rain streaks produce strong
        vertical edge responses distributed across the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.

        Returns
        -------
        bool
            True if vertical streak score exceeds threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._sobel_kernel_size)
        streak_score = float(np.mean(np.abs(sobel_v))) / 255.0
        return streak_score > self._streak_threshold

    def remove(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply temporal weighted average to remove rain streaks.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.

        Returns
        -------
        np.ndarray
            Rain-reduced BGR frame.
        """
        if frame is None or frame.size == 0:
            return frame

        self._frame_buffer.append(frame.astype(np.float32))

        n = len(self._frame_buffer)
        if n < 2:
            return frame

        # Use the last n weights (buffer may not be full yet)
        weights = self._WEIGHTS[-n:]
        weights = weights / weights.sum()  # re-normalise

        averaged = np.zeros_like(frame, dtype=np.float32)
        for w, f in zip(weights, self._frame_buffer):
            averaged += w * f

        return np.clip(averaged, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """Clear the frame buffer when switching videos."""
        self._frame_buffer.clear()
        logger.info("RainFilter reset")

    def __repr__(self) -> str:
        return (
            f"RainFilter(frames={self._num_frames}, "
            f"threshold={self._streak_threshold})"
        )
