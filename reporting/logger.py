import cv2
import json
import os
from datetime import datetime
import numpy as np


class EventLogger:
    """
    Logs accident events to a JSON file and saves snapshot images.
    """

    def __init__(self, config: dict):
        rcfg = config.get("reporting", {})

        self.snapshots_dir = rcfg.get("snapshots_dir", "output/snapshots")
        self.logs_dir      = rcfg.get("logs_dir", "output/logs")
        self.log_filename  = rcfg.get("log_filename", "events.json")
        self.save_on       = rcfg.get("save_snapshot_on", ["Moderate", "Severe", "Critical"])

        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.log_path = os.path.join(self.logs_dir, self.log_filename)
        self.events   = []

        # Start fresh for each pipeline run — overwrite old log
        self._write_log()

    def log(self, classified_event: dict, frame: "np.ndarray", frame_number: int):
        """
        Log one classified event. Save snapshot if severity warrants it.
        """
        timestamp  = datetime.now().isoformat()
        severity   = classified_event.get("severity", "Unknown")
        snapshot_path = None

        if severity in self.save_on and frame is not None:
            fname = f"{severity}_{timestamp.replace(':', '-')}_{frame_number}.jpg"
            snapshot_path = os.path.join(self.snapshots_dir, fname)
            cv2.imwrite(snapshot_path, frame)

        record = {
            "timestamp"    : timestamp,
            "frame_number" : frame_number,
            "track_id"     : classified_event.get("track_id"),
            "class_name"   : classified_event.get("class_name"),
            "severity"     : severity,
            "total_score"  : classified_event.get("total_score"),
            "signals"      : classified_event.get("all_signals", {}),
            "speed"        : classified_event.get("speed"),
            "avg_speed"    : classified_event.get("avg_speed"),
            "deviation_deg": classified_event.get("deviation_deg"),
            "bbox"         : classified_event.get("bbox"),
            "snapshot"     : snapshot_path
        }

        self.events.append(record)
        self._write_log()
        return record

    def _write_log(self):
        with open(self.log_path, "w") as f:
            json.dump(self.events, f, indent=2)

    def get_recent(self, n: int = 20) -> list:
        return self.events[-n:]