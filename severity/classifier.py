"""
Severity Classifier — Confidence-weighted, persistence-aware scoring.

Combines motion-based anomaly signals with YOLO-B hazard detections
to produce a severity level: Minor / Moderate / Severe / Critical.

Improvements over previous version:
  - YOLO-B signal scores are weighted by detection confidence
  - Persistence bonus: +2 if same track has been Moderate+ for 3 frames
  - Updated score map with higher weights for critical signals
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SeverityClassifier:
    """
    Multi-signal severity classifier with confidence weighting
    and temporal persistence bonus.
    """

    SEVERITY_LEVELS = ("Minor", "Moderate", "Severe", "Critical")

    def __init__(self, config: dict) -> None:
        scfg = config.get("severity", {})

        self._minor_max: int = scfg.get("minor_max", 3)
        self._moderate_max: int = scfg.get("moderate_max", 6)
        self._severe_max: int = scfg.get("severe_max", 10)

        scores = scfg.get("scores", {})
        self._score_map: Dict[str, float] = {
            "speed_drop_heavy": scores.get("speed_drop_heavy", 3),
            "speed_drop_moderate": scores.get("speed_drop_moderate", 1),
            "trajectory_deviation": scores.get("trajectory_deviation", 3),
            "vehicle_overlap": scores.get("vehicle_overlap", 3),
            "flow_incoherence": scores.get("flow_incoherence", 1),
            "motorcycle_track_lost": scores.get("motorcycle_track_lost", 4),
        }

        # Hazard signal base scores (multiplied by confidence)
        self._hazard_base: Dict[str, float] = {
            "crashed_vehicle": scores.get("crashed_vehicle_yolo", 4),
            "fire": scores.get("fire_detected", 5),
            "smoke": scores.get("smoke_detected", 3),
        }

        # Persistence tracking
        self._persistence_frames: int = scfg.get("persistence_bonus_frames", 3)
        self._persistence_bonus: float = scfg.get("persistence_bonus_score", 2)
        self._severity_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self._persistence_frames)
        )

        logger.info(
            "SeverityClassifier initialised: thresholds=[%d/%d/%d], "
            "persistence=%d frames",
            self._minor_max,
            self._moderate_max,
            self._severe_max,
            self._persistence_frames,
        )

    def classify(self, anomaly: dict, hazard_detections: List[dict]) -> dict:
        """
        Classify a single anomaly event with hazard context.

        Parameters
        ----------
        anomaly : dict
            Anomaly event from AnomalyDetector.
        hazard_detections : list[dict]
            Current YOLO-B detections.

        Returns
        -------
        dict
            Enriched event with severity, total_score, all_signals.
        """
        signals = anomaly.get("signals", {})
        tid = anomaly.get("track_id", -1)

        # ── Motion signal scoring ─────────────────────────────────
        motion_score = 0.0
        for signal_name in signals:
            if signal_name in self._score_map:
                motion_score += self._score_map[signal_name]

        # ── Hazard signal scoring (confidence-weighted) ───────────
        hazard_score, hazard_signals = self._score_hazards(hazard_detections)

        # ── Persistence bonus ─────────────────────────────────────
        persistence_score = self._compute_persistence(tid)

        # ── Total score and level ─────────────────────────────────
        total_score = motion_score + hazard_score + persistence_score
        level = self._score_to_level(total_score)

        # Update severity history for persistence tracking
        self._severity_history[tid].append(level)

        all_signals = {**signals, **hazard_signals}
        if persistence_score > 0:
            all_signals["persistence_bonus"] = True

        return {
            **anomaly,
            "severity": level,
            "total_score": round(total_score, 1),
            "motion_score": round(motion_score, 1),
            "hazard_score": round(hazard_score, 1),
            "persistence_score": round(persistence_score, 1),
            "all_signals": all_signals,
        }

    def _score_hazards(
        self, hazard_detections: List[dict]
    ) -> Tuple[float, dict]:
        """
        Score YOLO-B hazard detections with confidence weighting.

        Returns (total_score, hazard_signal_dict).
        """
        score = 0.0
        hazard_signals: Dict[str, bool] = {}

        for det in hazard_detections:
            name = det.get("class_name", "")
            conf = det.get("confidence", 0.5)

            if name in self._hazard_base:
                base = self._hazard_base[name]
                score += base * conf

                # Map to signal names
                signal_key = {
                    "crashed_vehicle": "crashed_vehicle_yolo",
                    "fire": "fire_detected",
                    "smoke": "smoke_detected",
                }.get(name)
                if signal_key:
                    hazard_signals[signal_key] = True

        return score, hazard_signals

    def _compute_persistence(self, tid: int) -> float:
        """
        Add persistence bonus if this track has been classified
        as Moderate or higher for the last N frames.
        """
        history = self._severity_history.get(tid)
        if history is None or len(history) < self._persistence_frames:
            return 0.0

        moderate_plus = {"Moderate", "Severe", "Critical"}
        if all(level in moderate_plus for level in history):
            return self._persistence_bonus

        return 0.0

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to severity level string."""
        if score <= self._minor_max:
            return "Minor"
        elif score <= self._moderate_max:
            return "Moderate"
        elif score <= self._severe_max:
            return "Severe"
        else:
            return "Critical"

    def reset(self) -> None:
        """Clear persistence history."""
        self._severity_history.clear()
        logger.info("SeverityClassifier reset")

    def __repr__(self) -> str:
        return (
            f"SeverityClassifier(thresholds="
            f"[{self._minor_max}/{self._moderate_max}/{self._severe_max}])"
        )