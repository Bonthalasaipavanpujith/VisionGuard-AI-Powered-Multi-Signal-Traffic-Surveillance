"""
ByteTrack + Kalman Filter + Lucas-Kanade Optical Flow tracker.

This is the most critical file in the pipeline. It implements:

1. ByteTrack two-step association (high-conf + low-conf matching)
   with Hungarian algorithm for optimal assignment.
2. Kalman Filter (cv2.KalmanFilter) for position prediction during
   lost frames.
3. Lucas-Kanade Optical Flow for per-vehicle velocity computation
   — this is MANDATORY per the base research paper.

ByteTrack handles identity assignment and track management.
LK optical flow computes the actual velocity vectors, speed values,
trajectory history, and flow incoherence scores for every confirmed
track. The anomaly detection math depends on these LK-computed values.
"""

import logging
from collections import Counter, deque
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class TrackState(IntEnum):
    """Track lifecycle states."""

    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2


class Track:
    """
    Single tracked object with full motion state.

    Holds ByteTrack identity, Kalman filter for position prediction,
    and LK optical flow data for velocity/incoherence computation.
    """

    __slots__ = (
        "track_id",
        "bbox",
        "class_votes",
        "state",
        "hits",
        "age",
        "kalman",
        "velocity",
        "speed",
        "incoherence",
        "speed_history",
        "velocity_history",
        "max_speed_ever",
        "first_seen_frame",
        "prev_points",
        "_history_len",
    )

    def __init__(
        self,
        track_id: int,
        bbox: List[int],
        class_name: str,
        frame_num: int,
        history_len: int = 30,
    ) -> None:
        self.track_id: int = track_id
        self.bbox: List[int] = list(bbox)
        self.class_votes: Counter = Counter({class_name: 1})
        self.state: TrackState = TrackState.TENTATIVE
        self.hits: int = 1
        self.age: int = 0
        self._history_len: int = history_len

        # Kalman filter: state [x, y, vx, vy], measurement [x, y]
        self.kalman: cv2.KalmanFilter = self._init_kalman(bbox)

        # Optical flow state
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.speed: float = 0.0
        self.incoherence: float = 0.0
        self.speed_history: deque = deque(maxlen=history_len)
        self.velocity_history: deque = deque(maxlen=history_len)
        self.max_speed_ever: float = 0.0
        self.first_seen_frame: int = frame_num
        self.prev_points: Optional[np.ndarray] = None

    @staticmethod
    def _init_kalman(bbox: List[int]) -> cv2.KalmanFilter:
        """Initialise a 4-state 2-measurement Kalman filter."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

        return kf

    @property
    def class_name(self) -> str:
        """Most frequently voted class label."""
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "unknown"

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def predict(self) -> np.ndarray:
        """Run Kalman prediction step. Returns predicted centroid."""
        return self.kalman.predict()

    def correct(self, cx: float, cy: float) -> None:
        """Run Kalman correction with measured centroid."""
        self.kalman.correct(np.array([[cx], [cy]], dtype=np.float32))

    def to_dict(self) -> dict:
        """Export track state as a dictionary for downstream modules."""
        return {
            "bbox": list(self.bbox),
            "class_name": self.class_name,
            "velocity": self.velocity,
            "speed": self.speed,
            "incoherence": self.incoherence,
            "speed_history": self.speed_history,
            "vel_history": self.velocity_history,
            "max_speed_ever": self.max_speed_ever,
            "age": self.age,
            "hits": self.hits,
            "confirmed": self.state == TrackState.CONFIRMED,
            "state": self.state,
            "first_seen_frame": self.first_seen_frame,
        }

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, cls={self.class_name}, "
            f"state={self.state.name}, speed={self.speed:.1f})"
        )


class ByteTracker:
    """
    ByteTrack multi-object tracker with Kalman prediction and
    Lucas-Kanade optical flow for velocity computation.

    Association uses two-step matching:
      Step A: High-confidence detections (>high_thresh) matched
              to existing tracks via IoU + Hungarian algorithm.
      Step B: Low-confidence detections matched to unmatched
              tracks from Step A.
    """

    def __init__(self, config: dict) -> None:
        tcfg = config.get("tracking", {})

        # ByteTrack thresholds
        self._high_thresh: float = tcfg.get("high_conf_threshold", 0.5)
        self._low_thresh: float = tcfg.get("low_conf_threshold", 0.3)
        self._iou_threshold: float = tcfg.get("iou_match_threshold", 0.7)
        self._max_lost: int = tcfg.get("max_lost_frames", 20)
        self._min_hits: int = tcfg.get("min_hits", 3)
        self._history_len: int = tcfg.get("history_length", 30)

        # LK optical flow parameters
        win = tcfg.get("lk_window_size", 15)
        lvl = tcfg.get("lk_max_level", 2)
        self._lk_params = dict(
            winSize=(win, win),
            maxLevel=lvl,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        self._corner_params = dict(
            maxCorners=tcfg.get("lk_max_corners", 100),
            qualityLevel=0.3,
            minDistance=5,
            blockSize=7,
        )
        self._min_lk_points: int = tcfg.get("min_lk_points", 3)
        self._alpha: float = tcfg.get("velocity_smoothing_alpha", 0.6)

        # State
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0

        logger.info(
            "ByteTracker initialised: high=%.2f, low=%.2f, "
            "max_lost=%d, min_hits=%d, iou=%.2f",
            self._high_thresh,
            self._low_thresh,
            self._max_lost,
            self._min_hits,
            self._iou_threshold,
        )

    # ── Public API ────────────────────────────────────────────────

    def update(
        self, frame: np.ndarray, detections: List[dict]
    ) -> Dict[int, dict]:
        """
        Main update method. Call every frame.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        detections : list[dict]
            Detections from VehicleDetector, each with bbox/class_name/confidence.

        Returns
        -------
        dict[int, dict]
            Confirmed tracks as {track_id: track_dict}.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_count += 1

        # ── Step 1: LK Optical Flow on existing confirmed tracks ──
        if self._prev_gray is not None:
            self._run_optical_flow(gray)

        # ── Step 2: Kalman predict for all tracks ─────────────────
        for track in self._tracks.values():
            track.predict()

        # ── Step 3: ByteTrack two-step association ────────────────
        high_dets = [d for d in detections if d["confidence"] >= self._high_thresh]
        low_dets = [
            d
            for d in detections
            if self._low_thresh <= d["confidence"] < self._high_thresh
        ]

        # Step A: match high-confidence detections
        active_tids = [
            tid
            for tid, t in self._tracks.items()
            if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)
        ]
        matched_a, unmatched_dets_a, unmatched_tracks_a = self._match(
            high_dets, active_tids
        )

        # Step B: match low-confidence detections to unmatched tracks
        matched_b, _, _ = self._match(low_dets, unmatched_tracks_a)

        # Combine matches
        all_matched_tids = set()
        for det_idx, tid in matched_a:
            self._update_track(tid, high_dets[det_idx])
            all_matched_tids.add(tid)
        for det_idx, tid in matched_b:
            self._update_track(tid, low_dets[det_idx])
            all_matched_tids.add(tid)

        # ── Step 4: Create new tracks from unmatched high-conf ────
        for det_idx in unmatched_dets_a:
            det = high_dets[det_idx]
            self._create_track(det, gray)

        # ── Step 5: Age unmatched tracks ──────────────────────────
        for tid, track in list(self._tracks.items()):
            if tid in all_matched_tids:
                continue

            track.age += 1
            if track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
            elif track.state == TrackState.LOST and track.age > self._max_lost:
                track.state = TrackState.TENTATIVE  # mark for deletion

            # Delete tracks that exceeded max lost time or never confirmed
            if track.age > self._max_lost:
                del self._tracks[tid]

        # ── Step 6: Extract feature points for next frame's LK ────
        # Extract for CONFIRMED and TENTATIVE so velocity builds early
        for track in self._tracks.values():
            if track.state in (TrackState.CONFIRMED, TrackState.TENTATIVE):
                track.prev_points = self._extract_points(gray, track.bbox)

        self._prev_gray = gray

        # Return confirmed tracks only
        return {
            tid: t.to_dict()
            for tid, t in self._tracks.items()
            if t.state == TrackState.CONFIRMED
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._next_id = 0
        self._prev_gray = None
        self._frame_count = 0
        logger.info("ByteTracker reset")

    # ── Matching ──────────────────────────────────────────────────

    def _match(
        self, detections: List[dict], track_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Hungarian algorithm matching between detections and tracks.

        Returns
        -------
        tuple
            (matched_pairs, unmatched_det_indices, unmatched_track_ids)
        """
        if not detections or not track_ids:
            return [], list(range(len(detections))), list(track_ids)

        # Build IoU cost matrix
        n_det = len(detections)
        n_trk = len(track_ids)
        cost = np.ones((n_det, n_trk), dtype=np.float32)

        for di, det in enumerate(detections):
            for ti, tid in enumerate(track_ids):
                iou = self._iou(det["bbox"], self._tracks[tid].bbox)
                cost[di, ti] = 1.0 - iou

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost)

        matched = []
        unmatched_dets = set(range(n_det))
        unmatched_trks = set(track_ids)

        for r, c in zip(row_indices, col_indices):
            if cost[r, c] > self._iou_threshold:
                continue  # reject poor matches
            tid = track_ids[c]
            matched.append((r, tid))
            unmatched_dets.discard(r)
            unmatched_trks.discard(tid)

        return matched, list(unmatched_dets), list(unmatched_trks)

    def _update_track(self, tid: int, det: dict) -> None:
        """Update a matched track with new detection."""
        track = self._tracks[tid]
        track.bbox = list(det["bbox"])
        track.class_votes[det["class_name"]] += 1
        track.age = 0
        track.hits += 1

        if track.hits >= self._min_hits:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

        # Kalman correction
        cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
        cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
        track.correct(cx, cy)

    def _create_track(self, det: dict, gray: np.ndarray) -> None:
        """Create a new tentative track from an unmatched detection."""
        track = Track(
            track_id=self._next_id,
            bbox=det["bbox"],
            class_name=det["class_name"],
            frame_num=self._frame_count,
            history_len=self._history_len,
        )
        track.prev_points = self._extract_points(gray, det["bbox"])
        self._tracks[self._next_id] = track
        self._next_id += 1

    # ── Lucas-Kanade Optical Flow ─────────────────────────────────

    def _run_optical_flow(self, gray: np.ndarray) -> None:
        """
        Run LK optical flow on all confirmed AND tentative tracks
        to compute velocity vectors, speed, and flow incoherence.

        Running on tentative tracks ensures velocity history is
        available as soon as a track gets confirmed.

        This is the MANDATORY component from the base research paper.
        """
        for track in self._tracks.values():
            # Run on both CONFIRMED and TENTATIVE (not LOST — no points)
            if track.state not in (TrackState.CONFIRMED, TrackState.TENTATIVE):
                continue

            pts = track.prev_points
            if pts is None or len(pts) < self._min_lk_points:
                # Fall back to Kalman velocity estimate
                self._fallback_velocity(track)
                continue

            # LK pyramidal optical flow
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None, **self._lk_params
            )

            if new_pts is None or status is None:
                self._fallback_velocity(track)
                continue

            mask = status.ravel() == 1
            good_new = new_pts[mask]
            good_old = pts[mask]

            if len(good_new) < self._min_lk_points:
                self._fallback_velocity(track)
                continue

            # Compute flow vectors
            flow_vectors = (good_new - good_old).reshape(-1, 2)

            raw_vx = float(np.mean(flow_vectors[:, 0]))
            raw_vy = float(np.mean(flow_vectors[:, 1]))
            raw_speed = float(np.sqrt(raw_vx * raw_vx + raw_vy * raw_vy))

            # Outlier clamp: threading may drop frames, causing huge
            # flow vectors between non-consecutive frames. Clamp to
            # a reasonable max (15 px/frame for CCTV footage).
            max_flow = 15.0
            if raw_speed > max_flow:
                scale = max_flow / raw_speed
                raw_vx *= scale
                raw_vy *= scale
                raw_speed = max_flow

            # EMA smoothing
            prev_vx, prev_vy = track.velocity
            vx = self._alpha * prev_vx + (1.0 - self._alpha) * raw_vx
            vy = self._alpha * prev_vy + (1.0 - self._alpha) * raw_vy

            speed = float(np.sqrt(vx * vx + vy * vy))

            # Flow incoherence: std of individual flow angles
            angles = np.degrees(
                np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
            )
            incoherence = float(np.std(angles)) if len(angles) > 1 else 0.0

            # Update track state
            track.velocity = (vx, vy)
            track.speed = speed
            track.incoherence = incoherence
            track.speed_history.append(speed)
            track.velocity_history.append((vx, vy))
            # Use RAW speed (not EMA) for max_speed_ever to capture true peaks
            track.max_speed_ever = max(track.max_speed_ever, raw_speed)

    def _fallback_velocity(self, track: Track) -> None:
        """
        Use bbox centroid displacement as velocity fallback when
        LK optical flow fails (too few feature points).
        """
        # Kalman state contains [x, y, vx, vy]
        state = track.kalman.statePost
        vx = float(state[2, 0])
        vy = float(state[3, 0])
        speed = float(np.sqrt(vx * vx + vy * vy))

        prev_vx, prev_vy = track.velocity
        vx = self._alpha * prev_vx + (1.0 - self._alpha) * vx
        vy = self._alpha * prev_vy + (1.0 - self._alpha) * vy
        speed = float(np.sqrt(vx * vx + vy * vy))

        track.velocity = (vx, vy)
        track.speed = speed
        track.incoherence = 0.0
        track.speed_history.append(speed)
        track.velocity_history.append((vx, vy))
        track.max_speed_ever = max(track.max_speed_ever, speed)

    # ── Feature Point Extraction ──────────────────────────────────

    def _extract_points(
        self, gray: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Extract Shi-Tomasi corners inside a bounding box for LK tracking.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1] - 1, x2)
        y2 = min(gray.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(roi, **self._corner_params)

        if pts is None:
            return None

        # Shift to full-frame coordinates
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _iou(b1: List[int], b2: List[int]) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def __repr__(self) -> str:
        n_confirmed = sum(
            1 for t in self._tracks.values() if t.state == TrackState.CONFIRMED
        )
        return (
            f"ByteTracker(tracks={len(self._tracks)}, "
            f"confirmed={n_confirmed})"
        )
