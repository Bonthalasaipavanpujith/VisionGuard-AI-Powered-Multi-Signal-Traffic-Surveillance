import requests
import time
import os
from datetime import datetime


class TelegramAlert:
    """
    Sends accident alerts to Telegram with snapshot image.
    Respects cooldown to avoid spamming.
    """

    SEVERITY_ORDER = {"Minor": 0, "Moderate": 1, "Severe": 2, "Critical": 3}

    SEVERITY_EMOJI = {
        "Minor"   : "🟡",
        "Moderate": "🟠",
        "Severe"  : "🔴",
        "Critical": "🚨"
    }

    def __init__(self, config: dict):
        acfg = config.get("alerts", {})

        self.enabled     = acfg.get("enable_telegram", False)
        self.bot_token   = acfg.get("bot_token", "")
        self.chat_id     = acfg.get("chat_id", "")
        self.min_severity = acfg.get("min_severity_to_alert", "Moderate")
        self.cooldown    = acfg.get("cooldown_seconds", 30)

        self._last_sent  = 0   # timestamp of last alert sent
        self._base_url   = f"https://api.telegram.org/bot{self.bot_token}"

    def _should_send(self, severity: str) -> bool:
        """Check severity threshold and cooldown."""
        if not self.enabled:
            return False
        if not self.bot_token or not self.chat_id:
            return False

        # check severity meets minimum threshold
        event_level = self.SEVERITY_ORDER.get(severity, 0)
        min_level   = self.SEVERITY_ORDER.get(self.min_severity, 1)
        if event_level < min_level:
            return False

        # check cooldown
        if time.time() - self._last_sent < self.cooldown:
            return False

        return True

    def _build_message(self, event: dict) -> str:
        severity  = event.get("severity", "Unknown")
        emoji     = self.SEVERITY_EMOJI.get(severity, "⚠️")
        timestamp = event.get("timestamp", datetime.now().isoformat())
        track_id  = event.get("track_id", "N/A")
        vehicle   = event.get("class_name", "unknown")
        score     = event.get("total_score", 0)
        signals   = list(event.get("signals", {}).keys())
        frame     = event.get("frame_number", "N/A")
        speed     = event.get("speed", 0)

        msg = (
            f"{emoji} *VisionGuard Alert*\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"🔺 *Severity:* {severity}\n"
            f"🚗 *Vehicle:* {vehicle} (Track #{track_id})\n"
            f"📊 *Score:* {score}\n"
            f"🎞 *Frame:* {frame}\n"
            f"💨 *Speed:* {speed:.1f} px/frame\n"
            f"⚡ *Signals:* {', '.join(signals) if signals else 'none'}\n"
            f"🕐 *Time:* {timestamp}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"_Sent by VisionGuard System_"
        )
        return msg

    def _send_text(self, message: str) -> bool:
        try:
            url  = f"{self._base_url}/sendMessage"
            data = {
                "chat_id"   : self.chat_id,
                "text"      : message,
                "parse_mode": "Markdown"
            }
            resp = requests.post(url, data=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram] Text send failed: {e}")
            return False

    def _send_photo(self, image_path: str, caption: str) -> bool:
        try:
            url = f"{self._base_url}/sendPhoto"
            with open(image_path, "rb") as img:
                resp = requests.post(
                    url,
                    data={
                        "chat_id"   : self.chat_id,
                        "caption"   : caption,
                        "parse_mode": "Markdown"
                    },
                    files={"photo": img},
                    timeout=15
                )
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram] Photo send failed: {e}")
            return False

    def send(self, event: dict):
        """
        Main method called from pipeline engine.
        Sends alert if severity threshold and cooldown conditions are met.
        """
        severity = event.get("severity", "Minor")

        if not self._should_send(severity):
            return

        message  = self._build_message(event)
        snapshot = event.get("snapshot")

        # send photo with caption if snapshot exists, else text only
        if snapshot and os.path.exists(snapshot):
            short_caption = (
                f"{self.SEVERITY_EMOJI.get(severity, '⚠️')} "
                f"*{severity} Accident Detected*\n"
                f"Score: {event.get('total_score', 0)} | "
                f"Frame: {event.get('frame_number', 'N/A')}"
            )
            success = self._send_photo(snapshot, short_caption)
            if success:
                # send full details as follow-up text
                self._send_text(message)
        else:
            success = self._send_text(message)

        if success:
            self._last_sent = time.time()
            print(f"[Telegram] Alert sent — {severity}")
        else:
            print(f"[Telegram] Alert failed to send")

    def test_connection(self) -> bool:
        """
        Call this once at startup to verify bot token and chat ID work.
        """
        try:
            msg = (
                "✅ *VisionGuard Connected*\n"
                "Alert system is active and working."
            )
            return self._send_text(msg)
        except Exception:
            return False