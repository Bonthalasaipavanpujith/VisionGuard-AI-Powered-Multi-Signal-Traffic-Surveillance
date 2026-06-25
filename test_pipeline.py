"""
VisionGuard Pipeline Benchmark — Per-stage timing validation.

Runs each pipeline stage independently and reports timing against
performance targets. Use this to verify the pipeline meets real-time
requirements on your hardware.

Usage:
    python test_pipeline.py --video video2.mp4
    python test_pipeline.py --video video2.mp4 --frames 100
"""

import argparse
import time
import sys

import cv2
import numpy as np
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def benchmark_motion_gate(config: dict, frames: list) -> float:
    from preprocessing.motion_gate import MotionGate

    gate = MotionGate(config)
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        gate.analyze(frame)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"  MOG2 Motion Gate:      {avg_ms:6.2f} ms/frame  (target: < 2ms)")
    return avg_ms


def benchmark_preprocessing(config: dict, frames: list) -> float:
    from preprocessing import AdaptivePreprocessor

    preprocessor = AdaptivePreprocessor(config)
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        preprocessor.process(frame)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"  Adaptive Preprocess:   {avg_ms:6.2f} ms/frame  (target: < 8ms)")
    return avg_ms


def benchmark_yoloa(config: dict, frames: list) -> float:
    from detection.vehicle_detector import VehicleDetector

    detector = VehicleDetector(config)
    times = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        detector.detect(frame, frame_num=i, force=True)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"  YOLO-A Detection:      {avg_ms:6.2f} ms/frame  (target: < 150ms)")
    return avg_ms


def benchmark_tracker(config: dict, frames: list, dets_list: list) -> float:
    from tracking.tracker import ByteTracker

    tracker = ByteTracker(config)
    times = []
    for frame, dets in zip(frames, dets_list):
        t0 = time.perf_counter()
        tracker.update(frame, dets)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"  ByteTrack + LK Flow:   {avg_ms:6.2f} ms/frame  (target: < 15ms)")
    return avg_ms


def benchmark_anomaly(config: dict, tracks_list: list) -> float:
    from anomaly.detector import AnomalyDetector

    detector = AnomalyDetector(config)
    times = []
    for i, tracks in enumerate(tracks_list):
        t0 = time.perf_counter()
        detector.detect(tracks, frame_num=i)
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"  Anomaly Detection:     {avg_ms:6.2f} ms/frame  (target: < 3ms)")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="VisionGuard Pipeline Benchmark")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to test")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  VisionGuard Pipeline Benchmark")
    print(f"  Video: {args.video}")
    print(f"  Frames: {args.frames}")
    print(f"{'='*60}\n")

    # Load frames
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    frames = []
    for _ in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 10:
        print(f"ERROR: Only {len(frames)} frames loaded, need at least 10")
        sys.exit(1)

    print(f"  Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})\n")
    print("  Stage Timings:")
    print("  " + "-" * 56)

    # 1. Motion gate
    t_motion = benchmark_motion_gate(config, frames)

    # 2. Full preprocessing
    t_preprocess = benchmark_preprocessing(config, frames)

    # 3. YOLO-A
    t_yoloa = benchmark_yoloa(config, frames[:min(20, len(frames))])

    # 4. Tracker (need detections first)
    from detection.vehicle_detector import VehicleDetector
    detector = VehicleDetector(config)
    dets_list = []
    for i, frame in enumerate(frames):
        dets = detector.detect(frame, frame_num=i, force=(i % 3 == 0))
        dets_list.append(dets)

    t_tracker = benchmark_tracker(config, frames, dets_list)

    # 5. Anomaly detector (need tracks)
    from tracking.tracker import ByteTracker
    tracker = ByteTracker(config)
    tracks_list = []
    for frame, dets in zip(frames, dets_list):
        tracks = tracker.update(frame, dets)
        tracks_list.append(tracks)

    t_anomaly = benchmark_anomaly(config, tracks_list)

    print("  " + "-" * 56)

    # Summary
    amortised_yoloa = t_yoloa / config.get("pipeline", {}).get("detect_interval", 3)
    total_amortised = t_motion + (t_preprocess - t_motion) + amortised_yoloa + t_tracker + t_anomaly
    target_fps = 1000.0 / total_amortised if total_amortised > 0 else 0

    print(f"\n  Amortised YOLO-A:      {amortised_yoloa:6.2f} ms/frame")
    print(f"  Total Amortised:       {total_amortised:6.2f} ms/frame")
    print(f"  Estimated FPS:         {target_fps:6.1f} FPS")
    print(f"\n  Target: > 14 FPS on laptop CPU")

    status = "✅ PASS" if target_fps >= 14 else "❌ FAIL"
    print(f"  Result: {status}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
