"""
Enhancer — Conditional CLAHE contrast enhancement.

CLAHE (Contrast Limited Adaptive Histogram Equalisation) enhances
local contrast without blowing out bright regions. Applied only to
the luminance channel (Y in YCrCb) to avoid colour distortion.

Only activated when frame contrast (std of grayscale) is below
threshold, avoiding unnecessary processing on well-lit frames.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Enhancer:
    """
    Conditional CLAHE contrast enhancement.

    The CLAHE object is created once in __init__ and reused for
    every frame — never recreated.
    """

    def __init__(self, config: dict) -> None:
        pcfg = config.get("preprocessing", {})

        clip_limit: float = pcfg.get("clahe_clip_limit", 2.0)
        tile_size: int = pcfg.get("clahe_tile_size", 8)
        self._contrast_threshold: float = pcfg.get("contrast_clahe_threshold", 35.0)

        # Created once, reused every frame
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )

        logger.info(
            "Enhancer initialised: clip=%.1f, tile=%d, threshold=%.1f",
            clip_limit,
            tile_size,
            self._contrast_threshold,
        )

    def needs_enhancement(self, frame: np.ndarray) -> bool:
        """
        Check whether the frame has low contrast and would benefit
        from CLAHE enhancement.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.

        Returns
        -------
        bool
            True if contrast (grayscale std) is below threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        return contrast < self._contrast_threshold

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the luminance channel only.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.

        Returns
        -------
        np.ndarray
            Contrast-enhanced BGR frame.
        """
        if frame is None or frame.size == 0:
            return frame

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y_enhanced = self._clahe.apply(y)

        enhanced_ycrcb = cv2.merge([y_enhanced, cr, cb])
        return cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)

    def reset(self) -> None:
        """No internal state to reset."""
        pass

    def __repr__(self) -> str:
        return f"Enhancer(threshold={self._contrast_threshold})"
