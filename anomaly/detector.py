"""
Anomaly Detector — Gated, deduplicated accident signal detection.

Implements five anomaly signals from the research paper:
  1. Speed Drop (heavy/moderate)
  2. Trajectory Deviation (dot product angle)
  3. Flow Incoherence (LK angle std)
  4. Vehicle Overlap (IoU-based)
  5. Motorcycle Disappearance

All signals are gated and require consecutive frame confirmation.
Per-track and global deduplication prevents alert storms.

Debug logging shows gate pass/fail for each track so threshold
tuning is transparent.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Gated anomaly detection with per-signal consecutive frame
    requirements and global rate limiting.
    """

    def __init__(self, config: dict) -> None:
        acfg = config.get("anomaly", {})

        # Hard gate thresholds
        self._min_frames: int = acfg.get("min_frames_before_detect", 8)
        self._min_max_speed: float = acfg.get("min_historical_max_speed", 1.5)
        self._event_cooldown_frames: int = acfg.get("event_cooldown_frames", 15)

        # Signal 1: Speed drop
        self._rolling_start: int = acfg.get("speed_drop_rolling_start", -10)
        self._rolling_end: int = acfg.get("speed_drop_rolling_end", -2)
        self._min_rolling_avg: float = acfg.get("min_rolling_avg_speed", 1.5)
        self._heavy_ratio: float = acfg.get("speed_drop_heavy_ratio", 0.30)
        self._moderate_ratio: float = acfg.get("speed_drop_moderate_ratio", 0.55)

        # Signal 2: Trajectory deviation
        self._traj_dev_deg: float = acfg.get("trajectory_deviation_deg", 45.0)

        # Signal 3: Flow incoherence
        self._incoherence_thresh: float = acfg.get("incoherence_threshold", 55.0)

        # Min speed for signals 2 & 3
        self._min_speed_signals: float = acfg.get("min_speed_for_signals", 1.5)

        # Consecutive frames required
        self._consecutive_req: int = acfg.get("consecutive_frames_required", 1)

        # Signal 4: Vehicle overlap
        self._overlap_iou: float = acfg.get("overlap_iou_threshold", 0.30)
        self._overlap_cooldown_frames: int = acfg.get("overlap_pair_cooldown", 20)

        # Signal 5: Motorcycle disappearance
        self._moto_min_ratio: float = acfg.get("motorcycle_min_ratio", 0.7)
        self._moto_history_frames: int = acfg.get("motorcycle_history_frames", 8)

        # Global rate limiting
        self._max_events_per_sec: int = acfg.get("global_events_per_second", 5)

        # ── Internal state ────────────────────────────────────────
        self._consecutive: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._event_cooldown: Dict[int, int] = {}
        self._overlap_cooldown: Dict[frozenset, int] = {}
        self._global_events: deque = deque(maxlen=self._max_events_per_sec * 2)
        self._moto_history: deque = deque(maxlen=self._moto_history_frames)
        self._moto_disappear_consec: int = 0

        # Debug: log gate stats every N frames
        self._debug_interval: int = 50
        self._last_debug_frame: int = 0

        logger.info(
            "AnomalyDetector initialised: gates=[%d frames, %.1f max_speed, %d cooldown], "
            "speed_thresholds=[heavy=%.2f, mod=%.2f, min_avg=%.1f], "
            "traj=%.0f°, incoherence=%.0f°, overlap_iou=%.2f",
            self._min_frames,
            self._min_max_speed,
            self._event_cooldown_frames,
            self._heavy_ratio,
            self._moderate_ratio,
            self._min_rolling_avg,
            self._traj_dev_deg,
            self._incoherence_thresh,
            self._overlap_iou,
        )

    def detect(
        self, tracks: Dict[int, dict], frame_num: int
    ) -> List[dict]:
        """
        Run gated anomaly detection on confirmed tracks.
        """
        anomalies: List[dict] = []
        track_list = list(tracks.items())
        anomaly_tids = set()

        # ── Debug logging ─────────────────────────────────────────
        if frame_num - self._last_debug_frame >= self._debug_interval:
            self._log_debug_stats(track_list, frame_num)
            self._last_debug_frame = frame_num

        # ── Update motorcycle presence history ────────────────────
        has_moto = any(
            t.get("class_name") == "motorcycle" for _, t in track_list
        )
        self._moto_history.append(has_moto)

        # ── Per-vehicle signal detection (signals 1-3) ────────────
        for tid, track in track_list:
            signals = self._check_track_signals(tid, track, frame_num)
            if not signals:
                continue

            if not self._rate_limit_ok():
                continue

            self._event_cooldown[tid] = frame_num
            self._global_events.append(time.monotonic())
            anomaly_tids.add(tid)

            anomalies.append(
                {
                    "track_id": tid,
                    "bbox": track["bbox"],
                    "class_name": track.get("class_name", "unknown"),
                    "signals": signals,
                    "speed": round(track.get("speed", 0), 2),
                    "avg_speed": round(self._rolling_avg(track), 2),
                    "deviation_deg": round(self._trajectory_angle(track), 1),
                    "incoherence": round(track.get("incoherence", 0), 1),
                    "overlap_with": None,
                }
            )

        # ── Signal 4: Vehicle overlap (pair-based) ────────────────
        # Merge with existing anomalies if same track already has signals
        overlap_anomalies = self._check_overlaps(
            track_list, frame_num, anomaly_tids
        )
        for ov in overlap_anomalies:
            existing = next(
                (a for a in anomalies if a["track_id"] == ov["track_id"]),
                None,
            )
            if existing:
                # Merge overlap signal into existing anomaly
                existing["signals"]["vehicle_overlap"] = True
                existing["overlap_with"] = ov.get("overlap_with")
            else:
                anomalies.append(ov)

        # ── Signal 5: Motorcycle disappeared ──────────────────────
        moto_anomaly = self._check_motorcycle_disappeared(
            track_list, frame_num
        )
        if moto_anomaly:
            anomalies.append(moto_anomaly)

        return anomalies

    def _log_debug_stats(
        self, track_list: list, frame_num: int
    ) -> None:
        """Log diagnostic info about all tracks for threshold tuning."""
        if not track_list:
            return

        for tid, track in track_list:
            speed_hist = track.get("speed_history")
            hist_len = len(speed_hist) if speed_hist else 0
            max_spd = track.get("max_speed_ever", 0)
            cur_spd = track.get("speed", 0)
            incoh = track.get("incoherence", 0)
            confirmed = track.get("confirmed", False)

            gate1 = hist_len >= self._min_frames
            gate2 = max_spd >= self._min_max_speed
            gate3 = confirmed
            last_fired = self._event_cooldown.get(tid, -999)
            gate4 = (frame_num - last_fired) >= self._event_cooldown_frames

            logger.debug(
                "Frame %d | Track %d (%s): speed=%.2f, max=%.2f, "
                "hist=%d, incoh=%.1f | gates=[G1:%s G2:%s G3:%s G4:%s]",
                frame_num,
                tid,
                track.get("class_name", "?"),
                cur_spd,
                max_spd,
                hist_len,
                incoh,
                "✓" if gate1 else "✗",
                "✓" if gate2 else "✗",
                "✓" if gate3 else "✗",
                "✓" if gate4 else "✗",
            )

    def _check_track_signals(
        self, tid: int, track: dict, frame_num: int
    ) -> Optional[dict]:
        """
        Check signals 1-3 with hard gates.
        Returns signal dict or None if no signals fire.
        """
        # ── Hard Gate 1: minimum history ──────────────────────────
        speed_hist = track.get("speed_history")
        if speed_hist is None or len(speed_hist) < self._min_frames:
            return None

        # ── Hard Gate 2: was vehicle ever moving? ─────────────────
        if track.get("max_speed_ever", 0) < self._min_max_speed:
            return None

        # ── Hard Gate 3: confirmed state ──────────────────────────
        if not track.get("confirmed", False):
            return None

        # ── Hard Gate 4: cooldown ─────────────────────────────────
        last_fired = self._event_cooldown.get(tid, -999)
        if (frame_num - last_fired) < self._event_cooldown_frames:
            return None

        signals = {}
        speed = track.get("speed", 0)

        # ── Signal 1: Speed drop ─────────────────────────────────
        self._check_speed_drop(tid, track, speed, signals)

        # ── Signal 2: Trajectory deviation ───────────────────────
        self._check_trajectory_deviation(tid, track, speed, signals)

        # ── Signal 3: Flow incoherence ───────────────────────────
        self._check_incoherence(tid, track, speed, signals)

        return signals if signals else None

    def _check_speed_drop(
        self, tid: int, track: dict, speed: float, signals: dict
    ) -> None:
        """Check for sudden speed drop (heavy or moderate)."""
        rolling_avg = self._rolling_avg(track)
        if rolling_avg < self._min_rolling_avg:
            self._consecutive[tid]["slow"] = 0
            return

        ratio = speed / (rolling_avg + 1e-6)

        if ratio < self._moderate_ratio:
            self._consecutive[tid]["slow"] += 1
        else:
            self._consecutive[tid]["slow"] = 0
            return

        if self._consecutive[tid]["slow"] < self._consecutive_req:
            return

        if ratio < self._heavy_ratio:
            signals["speed_drop_heavy"] = True
        else:
            signals["speed_drop_moderate"] = True

    def _check_trajectory_deviation(
        self, tid: int, track: dict, speed: float, signals: dict
    ) -> None:
        """Check for sudden direction change using dot product angle."""
        if speed < self._min_speed_signals:
            self._consecutive[tid]["deviation"] = 0
            return

        angle = self._trajectory_angle(track)
        if angle > self._traj_dev_deg:
            self._consecutive[tid]["deviation"] += 1
        else:
            self._consecutive[tid]["deviation"] = 0

        if self._consecutive[tid]["deviation"] >= self._consecutive_req:
            signals["trajectory_deviation"] = True

    def _check_incoherence(
        self, tid: int, track: dict, speed: float, signals: dict
    ) -> None:
        """Check for chaotic flow (high angle std = crash/spin)."""
        if speed < self._min_speed_signals:
            self._consecutive[tid]["incoherence"] = 0
            return

        incoherence = track.get("incoherence", 0)
        if incoherence > self._incoherence_thresh:
            self._consecutive[tid]["incoherence"] += 1
        else:
            self._consecutive[tid]["incoherence"] = 0

        if self._consecutive[tid]["incoherence"] >= self._consecutive_req:
            signals["flow_incoherence"] = True

    def _check_overlaps(
        self, track_list: list, frame_num: int, already_flagged: set
    ) -> List[dict]:
        """Check all pairs of confirmed tracks for high IoU overlap."""
        anomalies: List[dict] = []
        n = len(track_list)

        for i in range(n):
            for j in range(i + 1, n):
                tid_a, track_a = track_list[i]
                tid_b, track_b = track_list[j]

                pair_key = frozenset({tid_a, tid_b})
                last = self._overlap_cooldown.get(pair_key, -999)
                if (frame_num - last) < self._overlap_cooldown_frames:
                    continue

                iou = self._iou(track_a["bbox"], track_b["bbox"])
                if iou <= self._overlap_iou:
                    continue

                if not self._rate_limit_ok():
                    continue

                self._overlap_cooldown[pair_key] = frame_num
                self._global_events.append(time.monotonic())

                # Fire for both tracks
                for tid, track, other_tid in [
                    (tid_a, track_a, tid_b),
                    (tid_b, track_b, tid_a),
                ]:
                    # If already in anomaly list, will be merged upstream
                    if tid in already_flagged:
                        # Still create entry for merge
                        anomalies.append(
                            {
                                "track_id": tid,
                                "signals": {"vehicle_overlap": True},
                                "overlap_with": other_tid,
                            }
                        )
                        continue

                    prev = self._event_cooldown.get(tid, -999)
                    if (frame_num - prev) < self._event_cooldown_frames:
                        continue

                    self._event_cooldown[tid] = frame_num
                    anomalies.append(
                        {
                            "track_id": tid,
                            "bbox": track["bbox"],
                            "class_name": track.get("class_name", "unknown"),
                            "signals": {"vehicle_overlap": True},
                            "speed": round(track.get("speed", 0), 2),
                            "avg_speed": 0,
                            "deviation_deg": 0,
                            "incoherence": 0,
                            "overlap_with": other_tid,
                        }
                    )

        return anomalies

    def _check_motorcycle_disappeared(
        self, track_list: list, frame_num: int
    ) -> Optional[dict]:
        """Detect motorcycle track loss as accident signal."""
        if len(self._moto_history) < self._moto_history_frames:
            return None

        moto_ratio = sum(self._moto_history) / len(self._moto_history)
        has_moto_now = any(
            t.get("class_name") == "motorcycle" for _, t in track_list
        )

        if moto_ratio >= self._moto_min_ratio and not has_moto_now:
            self._moto_disappear_consec += 1
        else:
            self._moto_disappear_consec = 0

        if self._moto_disappear_consec < self._consecutive_req:
            return None

        if not self._rate_limit_ok():
            return None

        for tid, track in track_list:
            if track.get("class_name") in ("car", "truck", "bus"):
                prev = self._event_cooldown.get(tid, -999)
                if (frame_num - prev) < self._event_cooldown_frames:
                    continue

                self._event_cooldown[tid] = frame_num
                self._global_events.append(time.monotonic())
                self._moto_disappear_consec = 0

                return {
                    "track_id": tid,
                    "bbox": track["bbox"],
                    "class_name": track.get("class_name", "unknown"),
                    "signals": {"motorcycle_track_lost": True},
                    "speed": round(track.get("speed", 0), 2),
                    "avg_speed": 0,
                    "deviation_deg": 0,
                    "incoherence": 0,
                    "overlap_with": None,
                }

        return None

    # ── Utility methods ───────────────────────────────────────────

    def _rolling_avg(self, track: dict) -> float:
        """Rolling average speed excluding last 2 frames for stability."""
        hist = track.get("speed_history")
        if hist is None or len(hist) < 3:
            return 0.0

        hist_list = list(hist)
        end = self._rolling_end if len(hist_list) > abs(self._rolling_end) else None
        start = self._rolling_start if len(hist_list) >= abs(self._rolling_start) else 0
        segment = hist_list[start:end]
        return float(np.mean(segment)) if segment else 0.0

    def _trajectory_angle(self, track: dict) -> float:
        """Angle between average historical velocity and current velocity."""
        vel_hist = track.get("vel_history")
        if vel_hist is None or len(vel_hist) < 5:
            return 0.0

        history = list(vel_hist)[-10:]
        avg_vx = float(np.mean([v[0] for v in history]))
        avg_vy = float(np.mean([v[1] for v in history]))
        avg_mag = np.sqrt(avg_vx ** 2 + avg_vy ** 2)

        cur_vx, cur_vy = track.get("velocity", (0, 0))
        cur_mag = np.sqrt(cur_vx ** 2 + cur_vy ** 2)

        if avg_mag < 0.3 or cur_mag < 0.3:
            return 0.0

        dot = avg_vx * cur_vx + avg_vy * cur_vy
        cos_angle = np.clip(dot / (avg_mag * cur_mag), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _iou(b1: list, b2: list) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _rate_limit_ok(self) -> bool:
        """Check global event rate limit."""
        now = time.monotonic()
        recent = sum(1 for t in self._global_events if now - t < 1.0)
        return recent < self._max_events_per_sec

    def reset(self) -> None:
        """Reset all internal state."""
        self._consecutive.clear()
        self._event_cooldown.clear()
        self._overlap_cooldown.clear()
        self._global_events.clear()
        self._moto_history.clear()
        self._moto_disappear_consec = 0
        logger.info("AnomalyDetector reset")

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector(gates=[{self._min_frames}f, "
            f"{self._min_max_speed}spd, {self._event_cooldown_frames}cd])"
        )