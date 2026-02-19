#!/usr/bin/env python3
"""CLI entry point for running scrapers."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraping.iss_blog_scraper import ISSBlogScraper
from src.scraping.oral_history_scraper import OralHistoryScraper

DEFAULT_DB = PROJECT_ROOT / "data" / "astro_sentiment.db"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"
DEFAULT_EXPEDITIONS = PROJECT_ROOT / "data" / "expeditions.json"


def main():
    parser = argparse.ArgumentParser(description="Scrape NASA data sources")
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB), help="Path to SQLite database"
    )
    parser.add_argument(
        "--raw-dir", type=str, default=str(DEFAULT_RAW), help="Path to raw data cache"
    )
    parser.add_argument("--blog-only", action="store_true", help="Only scrape blog")
    parser.add_argument(
        "--oral-only", action="store_true", help="Only scrape oral histories"
    )
    parser.add_argument(
        "--mvp",
        action="store_true",
        help="MVP mode: scrape only recent blog posts (~100 pages)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override max listing pages for blog scraper",
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
            logging.FileHandler(PROJECT_ROOT / "data" / "scrape.log"),
        ],
    )

    scrape_blog = not args.oral_only
    scrape_oral = not args.blog_only

    if scrape_blog:
        print("=== ISS Blog Scraper ===")
        scraper = ISSBlogScraper(
            db_path=args.db,
            raw_dir=args.raw_dir,
            expeditions_path=str(DEFAULT_EXPEDITIONS),
        )
        scraper.run(max_pages=args.max_pages, mvp_mode=args.mvp)

    if scrape_oral:
        print("\n=== Oral History Scraper ===")
        scraper = OralHistoryScraper(db_path=args.db, raw_dir=args.raw_dir)
        scraper.run()

    print("\nDone!")


if __name__ == "__main__":
    main()
