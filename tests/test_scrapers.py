"""Tests for the scrapers: HTML parsing, PDF cleaning, segment splitting."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

# Mock pymupdf before importing oral_history_scraper
sys.modules.setdefault("pymupdf", MagicMock())

from src.scraping.iss_blog_scraper import ISSBlogScraper, fetch_page  # noqa: E402
from src.scraping.oral_history_scraper import (  # noqa: E402
    OralHistoryScraper,
    clean_pdf_text,
    split_segments,
)
from tests.conftest import SAMPLE_LISTING_HTML, SAMPLE_POST_HTML, SAMPLE_TRANSCRIPT_TEXT  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# ISS Blog Scraper
# ══════════════════════════════════════════════════════════════════════


class TestISSBlogListingParsing:
    """Test HTML parsing of listing pages without HTTP requests."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """A scraper instance with mocked DB and filesystem."""
        db_path = tmp_path / "test.db"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        exp_path = tmp_path / "expeditions.json"
        exp_path.write_text('[{"number":65,"name":"Exp 65","start_date":"2021-04-09","end_date":"2021-10-17","crew":["A"]}]')

        with patch("src.scraping.iss_blog_scraper.get_connection") as mock_conn, \
             patch("src.scraping.iss_blog_scraper.init_db"):
            mock_conn.return_value = MagicMock()
            s = ISSBlogScraper(db_path, raw_dir, exp_path)
        return s

    def test_extract_listing_meta(self, scraper):
        soup = BeautifulSoup(SAMPLE_LISTING_HTML, "html.parser")
        articles = soup.select("article")
        assert len(articles) == 2

        meta = scraper._extract_listing_meta(articles[0])
        assert meta is not None
        assert meta["url"] == "https://blogs.nasa.gov/spacestation/2021/06/15/science-day/"
        assert meta["title"] == "Science Day on Station"
        assert meta["author"] == "Mark Garcia"
        assert "2021-06-15" in meta["date_str"]

    def test_extract_listing_meta_second_article(self, scraper):
        soup = BeautifulSoup(SAMPLE_LISTING_HTML, "html.parser")
        articles = soup.select("article")
        meta = scraper._extract_listing_meta(articles[1])
        assert meta["title"] == "Spacewalk Preparations"

    def test_extract_listing_meta_no_link(self, scraper):
        """Article without a link returns None."""
        soup = BeautifulSoup("<article><p>No link here</p></article>", "html.parser")
        article = soup.select_one("article")
        assert scraper._extract_listing_meta(article) is None


class TestISSBlogPostParsing:
    """Test parsing of individual blog post pages."""

    @pytest.fixture
    def scraper(self, tmp_path):
        db_path = tmp_path / "test.db"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        exp_path = tmp_path / "expeditions.json"
        exp_path.write_text('[{"number":65,"name":"Exp 65","start_date":"2021-04-09","end_date":"2021-10-17","crew":["A"]}]')

        with patch("src.scraping.iss_blog_scraper.get_connection") as mock_conn, \
             patch("src.scraping.iss_blog_scraper.init_db"):
            mock_conn.return_value = MagicMock()
            s = ISSBlogScraper(db_path, raw_dir, exp_path)
        return s

    def test_scrape_full_post_from_cached_html(self, scraper):
        """Parse a cached HTML post without making HTTP requests."""
        url = "https://blogs.nasa.gov/spacestation/2021/06/15/science-day/"
        # Cache the HTML
        scraper._cache_html("post_science-day", SAMPLE_POST_HTML)

        post = scraper.scrape_full_post(url)
        assert post is not None
        assert post.title == "Science Day on Station"
        assert post.published_date == date(2021, 6, 15)
        assert "microgravity experiments" in post.text
        assert post.word_count > 0
        assert post.expedition_id == 65

    def test_scrape_full_post_extracts_author(self, scraper):
        scraper._cache_html("post_science-day", SAMPLE_POST_HTML)
        url = "https://blogs.nasa.gov/spacestation/2021/06/15/science-day/"
        post = scraper.scrape_full_post(url)
        assert post.author == "Mark Garcia"


class TestISSBlogCheckpointing:
    @pytest.fixture
    def scraper(self, tmp_path):
        db_path = tmp_path / "test.db"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        exp_path = tmp_path / "expeditions.json"
        exp_path.write_text('[{"number":1,"name":"Exp 1","start_date":"2000-10-31","end_date":"2001-03-21","crew":["A"]}]')

        with patch("src.scraping.iss_blog_scraper.get_connection") as mock_conn, \
             patch("src.scraping.iss_blog_scraper.init_db"):
            mock_conn.return_value = MagicMock()
            s = ISSBlogScraper(db_path, raw_dir, exp_path)
        return s

    def test_checkpoint_roundtrip(self, scraper):
        checkpoint = scraper._load_checkpoint()
        assert checkpoint["last_listing_page"] == 0

        checkpoint["last_listing_page"] = 42
        checkpoint["scraped_post_urls"] = ["https://example.com/a", "https://example.com/b"]
        scraper._save_checkpoint(checkpoint)

        loaded = scraper._load_checkpoint()
        assert loaded["last_listing_page"] == 42
        assert len(loaded["scraped_post_urls"]) == 2

    def test_html_cache_roundtrip(self, scraper):
        scraper._cache_html("test_page", "<html>hello</html>")
        cached = scraper._get_cached_html("test_page")
        assert cached == "<html>hello</html>"

    def test_uncached_returns_none(self, scraper):
        assert scraper._get_cached_html("nonexistent") is None


class TestDetectMaxPages:
    @pytest.fixture
    def scraper(self, tmp_path):
        db_path = tmp_path / "test.db"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        exp_path = tmp_path / "expeditions.json"
        exp_path.write_text('[{"number":1,"name":"Exp 1","start_date":"2000-10-31","end_date":"2001-03-21","crew":["A"]}]')

        with patch("src.scraping.iss_blog_scraper.get_connection") as mock_conn, \
             patch("src.scraping.iss_blog_scraper.init_db"):
            mock_conn.return_value = MagicMock()
            s = ISSBlogScraper(db_path, raw_dir, exp_path)
        return s

    def test_detect_max_pages_from_cache(self, scraper):
        scraper._cache_html("listing_0001", SAMPLE_LISTING_HTML)
        max_pages = scraper.detect_max_pages()
        assert max_pages == 50


# ══════════════════════════════════════════════════════════════════════
# Oral History Scraper
# ══════════════════════════════════════════════════════════════════════


class TestCleanPdfText:
    def test_removes_page_numbers(self):
        text = "Some text.\n  42  \nMore text."
        cleaned = clean_pdf_text(text)
        assert "42" not in cleaned.split("\n")

    def test_removes_nasa_headers(self):
        text = "NASA Johnson Space Center Oral History Project\nActual content here."
        cleaned = clean_pdf_text(text)
        assert "Oral History Project" not in cleaned
        assert "Actual content here" in cleaned

    def test_removes_edited_transcript_header(self):
        text = "Edited Oral History Transcript\nContent follows."
        cleaned = clean_pdf_text(text)
        assert "Edited Oral History Transcript" not in cleaned

    def test_collapses_blank_lines(self):
        text = "Line one.\n\n\n\n\nLine two."
        cleaned = clean_pdf_text(text)
        assert "\n\n\n" not in cleaned
        assert "Line one." in cleaned
        assert "Line two." in cleaned

    def test_fixes_broken_words(self):
        text = "astro-\nnaut"
        cleaned = clean_pdf_text(text)
        assert "astronaut" in cleaned

    def test_empty_input(self):
        assert clean_pdf_text("") == ""


class TestSplitSegments:
    def test_splits_into_qa_segments(self):
        segments = split_segments(SAMPLE_TRANSCRIPT_TEXT, "Wright")
        assert len(segments) > 0
        # Should have both interviewer and astronaut segments
        speakers = {s["speaker"] for s in segments}
        assert "interviewer" in speakers
        assert "astronaut" in speakers

    def test_interviewer_identified_correctly(self):
        segments = split_segments(SAMPLE_TRANSCRIPT_TEXT, "Wright")
        interviewer_segs = [s for s in segments if s["speaker"] == "interviewer"]
        assert len(interviewer_segs) >= 2  # WRIGHT speaks at least twice with questions

    def test_astronaut_identified_correctly(self):
        segments = split_segments(SAMPLE_TRANSCRIPT_TEXT, "Wright")
        astro_segs = [s for s in segments if s["speaker"] == "astronaut"]
        assert len(astro_segs) >= 2  # WHITSON speaks at least twice

    def test_segment_text_not_empty(self):
        segments = split_segments(SAMPLE_TRANSCRIPT_TEXT, "Wright")
        for seg in segments:
            assert len(seg["text"]) > 20

    def test_short_segments_filtered(self):
        text = "WRIGHT: Hi.\nWHITSON: Hello."
        segments = split_segments(text, "Wright")
        assert len(segments) == 0  # Both segments < 20 chars

    def test_empty_text(self):
        segments = split_segments("", "Wright")
        assert segments == []

    def test_no_speaker_labels(self):
        text = "Just a plain paragraph without any speaker labels or colons."
        segments = split_segments(text, "Wright")
        assert segments == []

    def test_whitespace_collapsed(self):
        text = "WHITSON: This   has   lots    of    spaces    and   is   a   longer  segment for  testing  purposes that exceeds twenty characters."
        segments = split_segments(text, "Wright")
        if segments:
            # Multiple spaces should be collapsed to single
            assert "   " not in segments[0]["text"]


class TestOralHistoryScraperInit:
    def test_creates_pdf_directory(self, tmp_path):
        with patch("src.scraping.oral_history_scraper.get_connection") as mock_conn, \
             patch("src.scraping.oral_history_scraper.init_db"):
            mock_conn.return_value = MagicMock()
            scraper = OralHistoryScraper(tmp_path / "test.db", tmp_path / "raw")

        assert (tmp_path / "raw" / "pdfs").exists()
