"""
Dehazer — Dark Channel Prior fog/haze removal.

Based on He et al., "Single Image Haze Removal Using Dark Channel Prior".
Only activated when the adaptive preprocessor detects foggy conditions
(mean_brightness > 160 AND dark_channel_mean > 0.08).

Optimised with patch_size=7 (was 15) for 4x speedup.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Dehazer:
    """
    Removes fog and haze from frames using Dark Channel Prior.

    The dark channel is the minimum pixel value across colour channels
    in a local patch. In hazy images this value is artificially raised
    by atmospheric scattering. We estimate and subtract the haze.
    """

    def __init__(self, config: dict) -> None:
        pcfg = config.get("preprocessing", {})

        self._patch_size: int = pcfg.get("dcp_patch_size", 7)
        self._omega: float = pcfg.get("dcp_omega", 0.95)
        self._t_min: float = pcfg.get("dcp_t_min", 0.1)

        # Pre-build the erosion kernel (created once)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self._patch_size, self._patch_size)
        )

        logger.info(
            "Dehazer initialised: patch=%d, omega=%.2f, t_min=%.2f",
            self._patch_size,
            self._omega,
            self._t_min,
        )

    def remove(self, frame: np.ndarray) -> np.ndarray:
        """
        Dehaze a BGR frame. Returns the frame unchanged if conditions
        indicate no significant haze.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.

        Returns
        -------
        np.ndarray
            Dehazed BGR image, or original if no haze detected.
        """
        if frame is None or frame.size == 0:
            return frame

        image_f = frame.astype(np.float64) / 255.0

        # ── Dark channel computation ─────────────────────────────
        dark_channel = self._dark_channel(image_f)

        # Early exit — scene already clear
        if np.mean(dark_channel) < 0.05:
            return frame

        # ── Atmospheric light estimation ─────────────────────────
        atmospheric_light = self._atmospheric_light(image_f, dark_channel)

        # ── Transmission map ─────────────────────────────────────
        transmission = self._transmission_map(image_f, atmospheric_light)

        # ── Scene recovery ───────────────────────────────────────
        recovered = self._recover_scene(frame, transmission, atmospheric_light * 255.0)

        return recovered

    def _dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Minimum across channels then minimum in local patch."""
        min_channel = np.min(image, axis=2)
        return cv2.erode(min_channel, self._kernel)

    def _atmospheric_light(
        self, image: np.ndarray, dark_channel: np.ndarray
    ) -> np.ndarray:
        """Estimate global atmospheric light from brightest dark-channel pixels."""
        num_pixels = dark_channel.size
        num_brightest = max(int(num_pixels * 0.001), 1)

        flat_dark = dark_channel.ravel()
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]

        flat_image = image.reshape(-1, 3)
        return np.mean(flat_image[indices], axis=0)

    def _transmission_map(
        self, image: np.ndarray, atmospheric_light: np.ndarray
    ) -> np.ndarray:
        """Estimate per-pixel transmission t(x)."""
        normalised = image / (atmospheric_light + 1e-6)
        dark = self._dark_channel(normalised)
        transmission = 1.0 - self._omega * dark
        return np.clip(transmission, self._t_min, 1.0).astype(np.float32)

    def _recover_scene(
        self,
        image: np.ndarray,
        transmission: np.ndarray,
        atmospheric_light: np.ndarray,
    ) -> np.ndarray:
        """Recover haze-free image using the atmospheric scattering model."""
        image_f = image.astype(np.float64)
        t = transmission[:, :, np.newaxis]
        recovered = (image_f - atmospheric_light) / t + atmospheric_light
        return np.clip(recovered, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """No internal state to reset."""
        pass

    def __repr__(self) -> str:
        return (
            f"Dehazer(patch={self._patch_size}, "
            f"omega={self._omega}, t_min={self._t_min})"
        )
