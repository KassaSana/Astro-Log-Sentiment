#!/usr/bin/env python3
"""CLI entry point for running the analysis pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.runner import run_analysis

DEFAULT_DB = PROJECT_ROOT / "data" / "astro_sentiment.db"


def main():
    parser = argparse.ArgumentParser(description="Run NLP analysis pipeline")
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB), help="Path to SQLite database"
    )
    parser.add_argument(
        "--skip-sentiment", action="store_true", help="Skip sentiment analysis"
    )
    parser.add_argument(
        "--skip-emotion", action="store_true", help="Skip emotion analysis"
    )
    parser.add_argument(
        "--skip-linguistic", action="store_true", help="Skip linguistic analysis"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model inference (cpu or cuda)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "data" / "analyze.log"),
        ],
    )

    print("=== Analysis Pipeline ===")
    run_analysis(
        db_path=args.db,
        skip_sentiment=args.skip_sentiment,
        skip_emotion=args.skip_emotion,
        skip_linguistic=args.skip_linguistic,
        device=args.device,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
