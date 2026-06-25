"""
Frame Reader — Threaded video frame capture.

Reads frames from cv2.VideoCapture in a background thread and
places them into a bounded queue. If the queue is full, the oldest
frame is dropped to ensure the display always gets the latest frame.

This decouples frame capture from frame processing, preventing
the video source from blocking the pipeline.
"""

import logging
import queue
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameReader:
    """
    Background-threaded frame reader with bounded queue.

    Starts a daemon thread that continuously reads frames from
    the video source and puts them into a queue. When the queue
    is full, frames are silently dropped.
    """

    def __init__(
        self,
        video_path: str,
        queue_size: int = 4,
        stop_event: threading.Event = None,
    ) -> None:
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event: threading.Event = stop_event or threading.Event()
        self._thread: threading.Thread = None
        self._video_path: str = video_path

        # Video properties (read once)
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Warm up: read first frame to verify source works
        ret, warmup_frame = self._cap.read()
        if ret:
            try:
                self._queue.put_nowait(warmup_frame)
            except queue.Full:
                pass
        else:
            raise RuntimeError(f"Cannot read first frame from: {video_path}")

        logger.info(
            "FrameReader ready: %s | %.1f FPS | %d frames | %dx%d",
            video_path,
            self.fps,
            self.total_frames,
            self.width,
            self.height,
        )

    def start(self) -> "FrameReader":
        """Start the background reader thread."""
        self._thread = threading.Thread(
            target=self._read_loop, name="FrameReader", daemon=True
        )
        self._thread.start()
        logger.info("FrameReader thread started")
        return self

    def _read_loop(self) -> None:
        """Continuous frame reading loop (runs in background thread)."""
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.info("FrameReader reached end of video")
                self._stop_event.set()
                break

            # Non-blocking put — drop frame if queue full
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # Drop this frame — always prefer latest
                pass

        self._cap.release()
        logger.info("FrameReader thread exited")

    def get_frame(self, timeout: float = 0.5):
        """
        Get the next frame from the queue.

        Parameters
        ----------
        timeout : float
            Seconds to wait before returning None.

        Returns
        -------
        np.ndarray or None
            BGR frame, or None if queue is empty / timed out.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """Check if the reader thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap.isOpened():
            self._cap.release()
        logger.info("FrameReader stopped")

    def reset(self) -> None:
        """Stop and release resources."""
        self.stop()

    def __repr__(self) -> str:
        return (
            f"FrameReader({self._video_path}, "
            f"fps={self.fps:.1f}, frames={self.total_frames})"
        )
