import argparse
import logging
import sys

from pipeline import VisionGuardEngine


def main():
    parser = argparse.ArgumentParser(description="VisionGuard Surveillance System")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without showing window (for servers)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = VisionGuardEngine(config_path=args.config)
    engine.run_video(args.video, show_window=not args.no_display)


if __name__ == "__main__":
    main()
