"""
Motion Gate — MOG2 Background Subtraction.

The single most important performance optimization in the pipeline.
On typical CCTV footage, 60-80% of frames have no significant motion.
By detecting this early, we skip ALL downstream processing (YOLO-A,
tracking, anomaly detection, YOLO-B) for static frames.

Uses cv2.createBackgroundSubtractorMOG2 which learns the background
model adaptively and produces a foreground mask per frame.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MotionResult:
    """Result of motion analysis for a single frame."""

    has_motion: bool = False
    motion_ratio: float = 0.0
    motion_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    motion_rois: List[Tuple[int, int, int, int]] = field(default_factory=list)
    frame_quality: float = 50.0

    def __repr__(self) -> str:
        return (
            f"MotionResult(has_motion={self.has_motion}, "
            f"motion_ratio={self.motion_ratio:.4f}, "
            f"rois={len(self.motion_rois)}, "
            f"quality={self.frame_quality:.1f})"
        )


class MotionGate:
    """
    MOG2-based motion gate that determines whether a frame contains
    significant motion worth processing through the full pipeline.

    When has_motion is False, the pipeline skips YOLO-A, tracking,
    anomaly detection, and YOLO-B entirely — returning the cached
    last annotated frame for display.
    """

    def __init__(self, config: dict) -> None:
        mcfg = config.get("motion_gate", {})

        self._history: int = mcfg.get("history", 200)
        self._var_threshold: int = mcfg.get("var_threshold", 50)
        self._min_motion_ratio: float = mcfg.get("min_motion_ratio", 0.002)
        self._min_contour_area: int = mcfg.get("min_contour_area", 500)

        # Create MOG2 subtractor — one-time allocation
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )

        # Morphological kernel — created once, reused every frame
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Pre-allocate Laplacian kernel reference for blur estimation
        self._warmup_frames: int = 0
        self._warmup_threshold: int = 30  # MOG2 needs ~30 frames to stabilise

        logger.info(
            "MotionGate initialised: history=%d, var_threshold=%d, "
            "min_motion_ratio=%.4f, min_contour_area=%d",
            self._history,
            self._var_threshold,
            self._min_motion_ratio,
            self._min_contour_area,
        )

    def analyze(self, frame: np.ndarray) -> MotionResult:
        """
        Analyse a BGR frame for significant motion.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from video capture.

        Returns
        -------
        MotionResult
            Dataclass with motion detection results and quality metrics.
        """
        if frame is None or frame.size == 0:
            return MotionResult()

        h, w = frame.shape[:2]
        frame_area = h * w

        # ── 1. Apply MOG2 to get foreground mask ─────────────────
        fg_mask = self._subtractor.apply(frame)

        # ── 2. Morphological cleanup ─────────────────────────────
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        # ── 3. Find contours and filter by area ──────────────────
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_motion_area = 0
        motion_rois: List[Tuple[int, int, int, int]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_contour_area:
                continue
            total_motion_area += area
            x, y, bw, bh = cv2.boundingRect(contour)
            motion_rois.append((x, y, x + bw, y + bh))

        # ── 4. Compute motion ratio ──────────────────────────────
        motion_ratio = total_motion_area / frame_area if frame_area > 0 else 0.0

        # ── 5. Determine if motion is significant ────────────────
        # During warmup, always report motion so MOG2 can learn
        self._warmup_frames += 1
        if self._warmup_frames <= self._warmup_threshold:
            has_motion = True
        else:
            has_motion = motion_ratio > self._min_motion_ratio

        # ── 6. Frame quality estimation ──────────────────────────
        frame_quality = self._estimate_quality(frame)

        return MotionResult(
            has_motion=has_motion,
            motion_ratio=round(motion_ratio, 6),
            motion_mask=cleaned,
            motion_rois=motion_rois,
            frame_quality=round(frame_quality, 1),
        )

    def _estimate_quality(self, frame: np.ndarray) -> float:
        """
        Estimate frame quality based on brightness spread and sharpness.

        Returns a score from 0 (very poor) to 100 (excellent).
        Uses brightness standard deviation and Laplacian variance as
        proxies for contrast and sharpness respectively.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness spread — low std = low contrast (foggy/washed out)
        brightness_std = float(np.std(gray))

        # Sharpness via Laplacian variance — low = blurry
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalise both to 0-50 range and sum
        contrast_score = min(brightness_std / 80.0 * 50.0, 50.0)
        sharpness_score = min(laplacian_var / 500.0 * 50.0, 50.0)

        return contrast_score + sharpness_score

    def reset(self) -> None:
        """Reset the background model and warmup counter."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )
        self._warmup_frames = 0
        logger.info("MotionGate reset")

    def __repr__(self) -> str:
        return (
            f"MotionGate(history={self._history}, "
            f"var_threshold={self._var_threshold}, "
            f"min_ratio={self._min_motion_ratio})"
        )
