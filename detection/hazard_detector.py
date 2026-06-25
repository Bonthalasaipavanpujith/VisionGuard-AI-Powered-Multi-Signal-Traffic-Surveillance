"""
Hazard Detector — YOLO-B wrapper with ROI-based conditional detection.

Only runs when anomalies are detected (not every frame).
Crops ROIs from anomaly track bboxes, runs YOLO-B on each crop,
and remaps coordinates back to the full frame.

Results are cached with a configurable TTL to avoid redundant inference.

Classes: crashed_vehicle(0), fire(1), smoke(2)
"""

import logging
from typing import List

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class HazardDetector:
    """
    YOLO-B hazard detector with ROI cropping and result caching.

    Only triggered when the anomaly detector produces events.
    Uses ROI-based detection for faster inference than full-frame.
    """

    CLASS_NAMES = {0: "crashed_vehicle", 1: "fire", 2: "smoke"}

    def __init__(self, config: dict) -> None:
        mcfg = config.get("models", {})
        hcfg = config.get("hazard_detector", {})

        model_path = mcfg.get("yolob", "models/yolob_best.pt")
        self._confidence: float = hcfg.get("confidence", 0.50)
        self._iou: float = hcfg.get("iou", 0.45)
        self._cache_ttl: int = hcfg.get("cache_ttl_frames", 5)
        self._roi_expand: int = hcfg.get("roi_expand_px", 50)

        self._model = YOLO(model_path)

        # Cache
        self._cache: List[dict] = []
        self._cache_frame: int = -999

        logger.info(
            "HazardDetector loaded: conf=%.2f, iou=%.2f, "
            "cache_ttl=%d, roi_expand=%d",
            self._confidence,
            self._iou,
            self._cache_ttl,
            self._roi_expand,
        )

    def detect(
        self,
        frame: np.ndarray,
        anomaly_tracks: list,
        current_frame: int,
    ) -> List[dict]:
        """
        Run YOLO-B detection conditionally.

        Only runs when anomaly_tracks is non-empty and cache has expired.
        Crops ROIs around anomaly bboxes for faster inference.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame.
        anomaly_tracks : list
            List of anomaly dicts from AnomalyDetector, each with 'bbox'.
        current_frame : int
            Current frame number.

        Returns
        -------
        list[dict]
            Hazard detections with bbox, class_id, class_name, confidence.
        """
        if frame is None or frame.size == 0:
            return self._cache

        # No anomalies → no need to run YOLO-B
        if not anomaly_tracks:
            return self._cache

        # Cache still valid
        if (current_frame - self._cache_frame) < self._cache_ttl:
            return self._cache

        h, w = frame.shape[:2]
        all_detections: List[dict] = []

        for anomaly in anomaly_tracks:
            bbox = anomaly.get("bbox", [0, 0, w, h])
            x1, y1, x2, y2 = bbox

            # Expand ROI
            rx1 = max(0, x1 - self._roi_expand)
            ry1 = max(0, y1 - self._roi_expand)
            rx2 = min(w, x2 + self._roi_expand)
            ry2 = min(h, y2 + self._roi_expand)

            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            results = self._model.predict(
                roi,
                conf=self._confidence,
                iou=self._iou,
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Remap to full frame coordinates
                    all_detections.append(
                        {
                            "bbox": [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1],
                            "class_id": cls_id,
                            "class_name": self.CLASS_NAMES.get(cls_id, "unknown"),
                            "confidence": round(conf, 3),
                        }
                    )

        self._cache = all_detections
        self._cache_frame = current_frame

        return all_detections

    def reset(self) -> None:
        """Clear cached detections."""
        self._cache = []
        self._cache_frame = -999
        logger.info("HazardDetector reset")

    def __repr__(self) -> str:
        return (
            f"HazardDetector(conf={self._confidence}, "
            f"cache_ttl={self._cache_ttl})"
        )