"""
Vehicle Detector — YOLO-A wrapper with interval-based caching.

Runs YOLO-A detection every DETECT_INTERVAL frames, caching results
between runs. Only runs if the motion gate reports motion. This
amortises the ~150ms YOLO inference cost across multiple frames.

Classes: car(0), truck(1), bus(2), motorcycle(3)
"""

import logging
from typing import List

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    YOLO-A vehicle detector with configurable detection interval
    and result caching.
    """

    CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}

    def __init__(self, config: dict) -> None:
        mcfg = config.get("models", {})
        dcfg = config.get("detection", {})
        pcfg = config.get("pipeline", {})

        model_path = mcfg.get("yoloa", "models/yoloa_best.pt")
        self._confidence: float = dcfg.get("yoloa_confidence", 0.30)
        self._iou: float = dcfg.get("iou_threshold", 0.35)
        self._max_det: int = dcfg.get("max_detections", 50)
        self._detect_interval: int = pcfg.get("detect_interval", 3)

        self._model = YOLO(model_path)

        # Cache
        self._cached_dets: List[dict] = []
        self._last_detect_frame: int = -999

        logger.info(
            "VehicleDetector loaded: conf=%.2f, iou=%.2f, "
            "interval=%d, max_det=%d",
            self._confidence,
            self._iou,
            self._detect_interval,
            self._max_det,
        )

    def detect(
        self, frame: np.ndarray, frame_num: int = 0, force: bool = False
    ) -> List[dict]:
        """
        Run YOLO-A detection, or return cached results if within interval.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.
        frame_num : int
            Current frame number for interval gating.
        force : bool
            Force detection regardless of interval.

        Returns
        -------
        list[dict]
            Detections with bbox, class_id, class_name, confidence.
        """
        if frame is None or frame.size == 0:
            return self._cached_dets

        # Interval check — return cache if not time to detect
        if not force and (frame_num - self._last_detect_frame) < self._detect_interval:
            return self._cached_dets

        h, w = frame.shape[:2]

        results = self._model.predict(
            frame,
            conf=self._confidence,
            iou=self._iou,
            imgsz=640,
            max_det=self._max_det,
            agnostic_nms=False,
            verbose=False,
        )

        detections: List[dict] = []
        max_w = w * 0.85
        max_h = h * 0.85

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                box_w = x2 - x1
                box_h = y2 - y1

                # Filter tiny boxes (noise)
                if box_w < 10 or box_h < 10:
                    continue

                # Filter unrealistically large boxes
                if box_w > max_w or box_h > max_h:
                    continue

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": cls_id,
                        "class_name": self.CLASS_NAMES.get(cls_id, "unknown"),
                        "confidence": round(conf, 3),
                    }
                )

        self._cached_dets = detections
        self._last_detect_frame = frame_num

        return detections

    def reset(self) -> None:
        """Clear cached detections."""
        self._cached_dets = []
        self._last_detect_frame = -999
        logger.info("VehicleDetector reset")

    def __repr__(self) -> str:
        return (
            f"VehicleDetector(conf={self._confidence}, "
            f"interval={self._detect_interval})"
        )