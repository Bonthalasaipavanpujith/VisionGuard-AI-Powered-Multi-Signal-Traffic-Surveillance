import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="VisionGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid;
    }
    .critical { border-color: #ff0000; }
    .severe   { border-color: #ff4444; }
    .moderate { border-color: #ff8800; }
    .minor    { border-color: #ffcc00; }
    .severity-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
    }
    .badge-Critical { background:#ff0000; color:white; }
    .badge-Severe   { background:#ff4444; color:white; }
    .badge-Moderate { background:#ff8800; color:white; }
    .badge-Minor    { background:#ffcc00; color:black; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────
SEVERITY_COLORS = {
    "Critical": "#ff0000",
    "Severe"  : "#ff4444",
    "Moderate": "#ff8800",
    "Minor"   : "#ffcc00"
}

SEVERITY_EMOJI = {
    "Critical": "🚨",
    "Severe"  : "🔴",
    "Moderate": "🟠",
    "Minor"   : "🟡"
}

def load_events(log_path: str) -> list:
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        try:
            return json.load(f)
        except Exception:
            return []

def severity_counts(events: list) -> dict:
    counts = {"Critical": 0, "Severe": 0, "Moderate": 0, "Minor": 0}
    for e in events:
        s = e.get("severity", "Minor")
        if s in counts:
            counts[s] += 1
    return counts

def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d %b %Y  %H:%M:%S")
    except Exception:
        return ts

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=60)
    st.title("VisionGuard")
    st.caption("Intelligent Surveillance System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📹 Live Detection", "📋 Event Log","⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.divider()

    log_path = st.text_input(
        "Events log path",
        value="output/logs/events.json"
    )

    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh"):
        st.rerun()

events = load_events(log_path)
counts = severity_counts(events)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — LIVE DETECTION
# ═══════════════════════════════════════════════════════════
if page == "📹 Live Detection":
    st.title("📹 Live Detection")
    st.caption("Run a video through VisionGuard and monitor detections in real time.")

    col1, col2 = st.columns([2, 1])

    with col1:
        video_path = st.text_input("Video file path", placeholder="C:/path/to/video.mp4")

        col_a, col_b = st.columns(2)
        with col_a:
            adaptive_on = st.toggle("Adaptive preprocessing", value=True)
        with col_b:
            detect_interval = st.slider("Detection interval", 1, 10, 3)

        run_btn = st.button("▶ Run VisionGuard", type="primary", use_container_width=True)

    with col2:
        st.markdown("### Live Stats")
        for severity, count in counts.items():
            color = SEVERITY_COLORS.get(severity, "#888")
            emoji = SEVERITY_EMOJI.get(severity, "")
            st.markdown(
                f"<div class='metric-card {severity.lower()}'>"
                f"<h2 style='color:{color};margin:0'>{count}</h2>"
                f"<p style='margin:0;color:#aaa'>{emoji} {severity}</p>"
                f"</div><br>",
                unsafe_allow_html=True
            )

    if run_btn and video_path:
        if not os.path.exists(video_path):
            st.error(f"Video not found: {video_path}")
        else:
            st.info("Starting VisionGuard pipeline...")

            # update config toggles
            import yaml
            config_path = "config/config.yaml"
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                cfg["preprocessing"]["adaptive"] = adaptive_on
                cfg["pipeline"]["detect_interval"] = detect_interval
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f)

            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                from pipeline import VisionGuardEngine
                engine = VisionGuardEngine(config_path=config_path)

                cap = cv2.VideoCapture(video_path)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (1280, 720))
                    out_frame, events_this_frame = engine.process_frame(frame)

                    # show every 5th frame in dashboard
                    if frame_count % 5 == 0:
                        rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb, width=1280)

                    if events_this_frame:
                        for e in events_this_frame:
                            sev = e.get("severity", "")
                            emoji = SEVERITY_EMOJI.get(sev, "")
                            status_placeholder.warning(
                                f"{emoji} {sev} event detected — "
                                f"Score: {e.get('total_score')} | "
                                f"Signals: {list(e.get('all_signals', {}).keys())}"
                            )

                    frame_count += 1

                cap.release()
                st.success(f"Processing complete. {frame_count} frames analysed.")

            except Exception as ex:
                st.error(f"Pipeline error: {ex}")
                st.exception(ex)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — EVENT LOG
# ═══════════════════════════════════════════════════════════
elif page == "📋 Event Log":
    st.title("📋 Event Log")

    if not events:
        st.info("No events logged yet. Run a video to generate events.")
    else:
        # filters
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_filter = st.multiselect(
                "Filter by severity",
                ["Critical", "Severe", "Moderate", "Minor"],
                default=["Critical", "Severe", "Moderate", "Minor"]
            )
        with col2:
            vehicle_filter = st.multiselect(
                "Filter by vehicle",
                list({e.get("class_name", "unknown") for e in events}),
                default=list({e.get("class_name", "unknown") for e in events})
            )
        with col3:
            sort_order = st.selectbox("Sort", ["Newest first", "Oldest first", "Severity (high→low)"])

        filtered = [
            e for e in events
            if e.get("severity") in severity_filter
            and e.get("class_name") in vehicle_filter
        ]

        if sort_order == "Newest first":
            filtered = list(reversed(filtered))
        elif sort_order == "Severity (high→low)":
            order = {"Critical": 0, "Severe": 1, "Moderate": 2, "Minor": 3}
            filtered.sort(key=lambda x: order.get(x.get("severity", "Minor"), 3))

        st.caption(f"Showing {len(filtered)} of {len(events)} events")
        st.divider()

        for event in filtered:
            severity = event.get("severity", "Unknown")
            emoji    = SEVERITY_EMOJI.get(severity, "⚠️")
            color    = SEVERITY_COLORS.get(severity, "#888")

            with st.expander(
                f"{emoji} {severity}  |  "
                f"{format_timestamp(event.get('timestamp', ''))}  |  "
                f"Track #{event.get('track_id')}  |  "
                f"{event.get('class_name', 'unknown')}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Severity:** "
                                f"<span style='color:{color}'>{severity}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"**Total score:** {event.get('total_score', 0)}")
                    st.markdown(f"**Motion score:** {event.get('motion_score', 0)}")
                    st.markdown(f"**Hazard score:** {event.get('hazard_score', 0)}")

                with col2:
                    st.markdown(f"**Frame:** {event.get('frame_number', 'N/A')}")
                    st.markdown(f"**Speed:** {event.get('speed', 0)} px/frame")
                    st.markdown(f"**Avg speed:** {event.get('avg_speed', 0)} px/frame")
                    st.markdown(f"**Direction deviation:** {event.get('deviation_deg', 0)}°")

                with col3:
                    signals = event.get("signals", {})
                    if signals:
                        st.markdown("**Triggered signals:**")
                        for sig in signals:
                            st.markdown(f"- `{sig}`")
                    else:
                        st.markdown("**Signals:** none")

                snapshot = event.get("snapshot")
                if snapshot and os.path.exists(snapshot):
                    st.image(snapshot, caption="Evidence snapshot", width=400)

# ═══════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS (TEMPORARILY DISABLED)
# ═══════════════════════════════════════════════════════════
# Analytics page disabled temporarily
# elif page == "📊 Analytics":
#     st.title("📊 Analytics")
#     st.info("Analytics page is temporarily disabled. Use Event Log for event details.")

# ═══════════════════════════════════════════════════════════
# PAGE 4 — SETTINGS
# ═══════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    import yaml
    config_path = "config/config.yaml"

    if not os.path.exists(config_path):
        st.error("config/config.yaml not found.")
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        st.subheader("Detection thresholds")
        col1, col2 = st.columns(2)
        with col1:
            cfg["detection"]["yoloa_confidence"] = st.slider(
                "YOLO-A confidence", 0.1, 0.9,
                float(cfg["detection"]["yoloa_confidence"]), 0.05
            )
        with col2:
            cfg["detection"]["yolob_confidence"] = st.slider(
                "YOLO-B confidence", 0.1, 0.9,
                float(cfg["detection"]["yolob_confidence"]), 0.05
            )

        st.subheader("Anomaly detection")
        col1, col2 = st.columns(2)
        with col1:
            cfg["anomaly"]["speed_drop_heavy_ratio"] = st.slider(
                "Speed drop heavy ratio", 0.05, 0.5,
                float(cfg["anomaly"].get("speed_drop_heavy_ratio", 0.25)), 0.05
            )
        with col2:
            cfg["anomaly"]["trajectory_deviation_deg"] = st.slider(
                "Trajectory deviation (degrees)", 10, 90,
                int(cfg["anomaly"].get("trajectory_deviation_deg", 50)), 5
            )

        st.subheader("Telegram alerts")
        cfg["alerts"]["enable_telegram"] = st.toggle(
            "Enable Telegram alerts",
            value=cfg["alerts"].get("enable_telegram", False)
        )
        if cfg["alerts"]["enable_telegram"]:
            cfg["alerts"]["bot_token"] = st.text_input(
                "Bot token",
                value=cfg["alerts"].get("bot_token", ""),
                type="password"
            )
            cfg["alerts"]["chat_id"] = st.text_input(
                "Chat ID",
                value=cfg["alerts"].get("chat_id", "")
            )

        if st.button("💾 Save settings", type="primary"):
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            st.success("Settings saved.")