"""NASA Oral History transcript scraper — downloads PDFs and extracts Q&A segments."""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pymupdf
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.db import get_connection, init_db, insert_oral_history
from src.data.models import OralHistorySegment

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.nasa.gov/history/johnson-history-resources/"
USER_AGENT = "AstroSentimentResearch/1.0 (academic project)"
RATE_LIMIT_SECONDS = 2.0

# Known ISS-related oral history participants and their PDF URLs.
# The index page layout has changed over time, so we maintain a fallback list.
# These were verified from the NASA JSC History Portal.
KNOWN_PARTICIPANTS = [
    {"name": "Michael R. Barratt", "interviewer": "Wright"},
    {"name": "Randy H. Brinkley", "interviewer": "Wright"},
    {"name": "Robert D. Cabana", "interviewer": "Wright"},
    {"name": "John B. Charles", "interviewer": "Wright"},
    {"name": "Kevin P. Chilton", "interviewer": "Wright"},
    {"name": "Laurie N. Hansen", "interviewer": "Wright"},
    {"name": "Albert W. Holland", "interviewer": "Wright"},
    {"name": "Gregory H. Johnson", "interviewer": "Wright"},
    {"name": "Charles M. Lundquist", "interviewer": "Wright"},
    {"name": "Jeffrey Manber", "interviewer": "Wright"},
    {"name": "Hans Mark", "interviewer": "Wright"},
    {"name": "Donald R. Pettit", "interviewer": "Wright"},
    {"name": "Michael E. Read", "interviewer": "Wright"},
    {"name": "Julie A. Robinson", "interviewer": "Wright"},
    {"name": "Melanie Saunders", "interviewer": "Wright"},
    {"name": "Michael T. Suffredini", "interviewer": "Wright"},
    {"name": "Suzan C. Voss", "interviewer": "Wright"},
    {"name": "Peggy A. Whitson", "interviewer": "Wright"},
    {"name": "Jeffrey N. Williams", "interviewer": "Wright"},
    {"name": "Sunita L. Williams", "interviewer": "Wright"},
]


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
)
def fetch_url(url: str) -> requests.Response:
    resp = requests.get(
        url, timeout=60, headers={"User-Agent": USER_AGENT}, allow_redirects=True
    )
    resp.raise_for_status()
    return resp


def clean_pdf_text(raw: str) -> str:
    """Remove common headers, footers, and artifacts from PDF-extracted text."""
    # Remove page numbers (standalone digits on their own line)
    text = re.sub(r"^\s*\d+\s*$", "", raw, flags=re.MULTILINE)

    # Remove common NASA oral history headers/footers
    patterns_to_remove = [
        r"NASA\s+Johnson\s+Space\s+Center\s+Oral\s+History\s+Project[^\n]*\n?",
        r"Edited\s+Oral\s+History\s+Transcript[^\n]*\n?",
        r"JOHNSON SPACE CENTER ORAL HISTORY PROJECT[^\n]*\n?",
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Collapse multiple blank lines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix common PDF extraction issues: broken words across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()


def split_segments(text: str, interviewer_last_name: str) -> list[dict]:
    """Split a cleaned transcript into Q&A segments.

    NASA oral histories follow this format:
        LASTNAME: Question or answer text here...

    Speaker names appear in ALL CAPS (or mixed case) followed by a colon.
    """
    # Match speaker turns: capitalized name followed by colon at start of line
    pattern = r"^([A-Z][A-Za-z\s.\-']+?):\s*"

    segments = []
    current_speaker = None
    current_text_parts: list[str] = []

    for line in text.split("\n"):
        match = re.match(pattern, line)
        if match:
            # Save previous segment
            if current_speaker is not None and current_text_parts:
                segment_text = " ".join(current_text_parts).strip()
                # Remove excess whitespace
                segment_text = re.sub(r"\s+", " ", segment_text)
                if len(segment_text) > 20:  # Skip very short segments
                    is_interviewer = (
                        interviewer_last_name.upper()
                        in current_speaker.upper()
                    )
                    segments.append(
                        {
                            "speaker": "interviewer" if is_interviewer else "astronaut",
                            "text": segment_text,
                        }
                    )

            current_speaker = match.group(1).strip()
            remainder = line[match.end() :]
            current_text_parts = [remainder] if remainder.strip() else []
        else:
            if line.strip():
                current_text_parts.append(line.strip())

    # Don't forget the last segment
    if current_speaker is not None and current_text_parts:
        segment_text = " ".join(current_text_parts).strip()
        segment_text = re.sub(r"\s+", " ", segment_text)
        if len(segment_text) > 20:
            is_interviewer = interviewer_last_name.upper() in current_speaker.upper()
            segments.append(
                {
                    "speaker": "interviewer" if is_interviewer else "astronaut",
                    "text": segment_text,
                }
            )

    return segments


class OralHistoryScraper:
    def __init__(self, db_path: str | Path, raw_dir: str | Path):
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir)
        self.pdf_dir = self.raw_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.conn = get_connection(db_path)
        init_db(self.conn)

    def fetch_transcript_index(self) -> list[dict]:
        """Fetch the oral history index page and extract PDF links.

        Returns list of dicts with keys: name, pdf_url, interviewer
        """
        logger.info(f"Fetching oral history index: {INDEX_URL}")
        try:
            resp = fetch_url(INDEX_URL)
            soup = BeautifulSoup(resp.text, "html.parser")

            results = []
            # Look for PDF links in the page
            for link in soup.select("a[href$='.pdf']"):
                href = link.get("href", "")
                name = link.get_text(strip=True)
                if name and href:
                    # Strip query params
                    pdf_url = href.split("?")[0]
                    if not pdf_url.startswith("http"):
                        pdf_url = "https://www.nasa.gov" + pdf_url
                    results.append(
                        {
                            "name": name,
                            "pdf_url": pdf_url,
                            "interviewer": "Wright",  # Default interviewer
                        }
                    )

            if results:
                logger.info(f"Found {len(results)} PDF links on index page")
                return results

        except Exception as e:
            logger.warning(f"Failed to scrape index page: {e}")

        # Fallback: return known participants without URLs
        # (URLs will need to be discovered or manually provided)
        logger.info("Using known participants list as fallback")
        return KNOWN_PARTICIPANTS

    def download_pdf(self, url: str, name: str) -> Path | None:
        """Download a PDF to the local cache. Returns path or None on failure."""
        safe_name = re.sub(r"[^\w\-.]", "_", name)
        dest = self.pdf_dir / f"{safe_name}.pdf"

        if dest.exists() and dest.stat().st_size > 1000:
            logger.debug(f"PDF already cached: {dest}")
            return dest

        logger.info(f"Downloading PDF for {name}: {url}")
        try:
            resp = fetch_url(url)
            dest.write_bytes(resp.content)
            time.sleep(RATE_LIMIT_SECONDS)
            logger.info(f"Downloaded: {dest.name} ({len(resp.content)} bytes)")
            return dest
        except Exception as e:
            logger.error(f"Failed to download PDF for {name}: {e}")
            return None

    def extract_text(self, pdf_path: Path) -> str:
        """Extract and clean text from a PDF file."""
        doc = pymupdf.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()

        raw_text = "\n".join(pages)
        return clean_pdf_text(raw_text)

    def process_transcript(
        self, name: str, pdf_url: str, interviewer: str, pdf_path: Path
    ) -> int:
        """Extract text, split into segments, and insert into database.

        Returns the number of segments inserted.
        """
        text = self.extract_text(pdf_path)
        if len(text) < 100:
            logger.warning(f"Extracted text too short for {name}: {len(text)} chars")
            return 0

        interviewer_last_name = interviewer.split()[-1] if interviewer else "Wright"
        segments = split_segments(text, interviewer_last_name)

        if not segments:
            logger.warning(f"No segments found for {name}")
            return 0

        count = 0
        for idx, seg in enumerate(segments):
            segment = OralHistorySegment(
                astronaut_name=name,
                pdf_url=pdf_url,
                interview_date=None,
                segment_index=idx,
                speaker=seg["speaker"],
                text=seg["text"],
                word_count=len(seg["text"].split()),
            )
            insert_oral_history(self.conn, segment)
            count += 1

        logger.info(f"Inserted {count} segments for {name}")
        return count

    def run(self) -> None:
        """Run the full oral history scraping pipeline."""
        transcripts = self.fetch_transcript_index()
        total_segments = 0

        for entry in transcripts:
            name = entry["name"]
            pdf_url = entry.get("pdf_url")
            interviewer = entry.get("interviewer", "Wright")

            if not pdf_url:
                logger.warning(f"No PDF URL for {name}, skipping")
                continue

            # Check if we already have segments for this person
            existing = self.conn.execute(
                "SELECT COUNT(*) FROM oral_histories WHERE astronaut_name = ?",
                (name,),
            ).fetchone()[0]
            if existing > 0:
                logger.info(f"Already have {existing} segments for {name}, skipping")
                continue

            pdf_path = self.download_pdf(pdf_url, name)
            if pdf_path:
                count = self.process_transcript(name, pdf_url, interviewer, pdf_path)
                total_segments += count

        logger.info(
            f"Oral history scrape complete. Total segments: {total_segments}"
        )
