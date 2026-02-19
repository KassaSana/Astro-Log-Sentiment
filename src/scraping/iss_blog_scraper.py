"""ISS Blog scraper with pagination, checkpointing, and rate limiting."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.data.db import (
    get_connection,
    init_db,
    insert_blog_post,
    load_expeditions,
    map_date_to_expedition,
)
from src.data.models import BlogPost

logger = logging.getLogger(__name__)

BASE_URL = "https://blogs.nasa.gov/spacestation"
LISTING_URL = BASE_URL + "/page/{page_num}/"
USER_AGENT = "AstroSentimentResearch/1.0 (academic project)"
RATE_LIMIT_SECONDS = 1.5


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
)
def fetch_page(url: str) -> requests.Response:
    """Fetch a URL with retries and a polite User-Agent."""
    resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp


class ISSBlogScraper:
    def __init__(
        self,
        db_path: str | Path,
        raw_dir: str | Path,
        expeditions_path: str | Path,
    ):
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir)
        self.html_dir = self.raw_dir / "html"
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.raw_dir / "blog_scrape_checkpoint.json"

        self.expeditions = load_expeditions(expeditions_path)
        self.conn = get_connection(db_path)
        init_db(self.conn)

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return {
            "last_listing_page": 0,
            "scraped_post_urls": [],
            "started_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["updated_at"] = datetime.utcnow().isoformat()
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _get_cached_html(self, cache_key: str) -> str | None:
        cache_file = self.html_dir / f"{cache_key}.html"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def _cache_html(self, cache_key: str, html: str) -> None:
        cache_file = self.html_dir / f"{cache_key}.html"
        cache_file.write_text(html, encoding="utf-8")

    def scrape_listing_page(self, page_num: int) -> list[dict]:
        """Scrape a single listing page and return post metadata.

        Returns list of dicts with keys: url, title, author, date_str
        """
        cache_key = f"listing_{page_num:04d}"
        html = self._get_cached_html(cache_key)

        if html is None:
            url = LISTING_URL.format(page_num=page_num)
            logger.info(f"Fetching listing page {page_num}: {url}")
            resp = fetch_page(url)
            html = resp.text
            self._cache_html(cache_key, html)
            time.sleep(RATE_LIMIT_SECONDS)

        soup = BeautifulSoup(html, "html.parser")
        posts = []

        # Try multiple selectors for WordPress blog layouts
        articles = soup.select("article")
        if not articles:
            articles = soup.select(".post")
        if not articles:
            articles = soup.select(".entry")

        for article in articles:
            post_meta = self._extract_listing_meta(article)
            if post_meta:
                posts.append(post_meta)

        logger.info(f"Page {page_num}: found {len(posts)} posts")
        return posts

    def _extract_listing_meta(self, article) -> dict | None:
        """Extract post URL, title, author, and date from a listing article element."""
        # Find the post link
        link = article.select_one("h2 a, h1 a, .entry-title a")
        if not link or not link.get("href"):
            return None

        url = link["href"].strip()
        title = link.get_text(strip=True)

        # Find the date
        date_str = None
        time_el = article.select_one("time")
        if time_el:
            date_str = time_el.get("datetime", time_el.get_text(strip=True))
        else:
            date_span = article.select_one(".posted-on, .entry-date, .date")
            if date_span:
                date_str = date_span.get_text(strip=True)

        # Find the author
        author = None
        author_el = article.select_one(".byline a, .author a, .entry-author a")
        if author_el:
            author = author_el.get_text(strip=True)

        return {"url": url, "title": title, "author": author, "date_str": date_str}

    def scrape_full_post(self, url: str) -> BlogPost | None:
        """Fetch and parse a full blog post page."""
        # Create a safe cache key from the URL slug
        slug = url.rstrip("/").split("/")[-1]
        cache_key = f"post_{slug[:80]}"
        html = self._get_cached_html(cache_key)

        if html is None:
            logger.info(f"Fetching post: {url}")
            try:
                resp = fetch_page(url)
                html = resp.text
                self._cache_html(cache_key, html)
                time.sleep(RATE_LIMIT_SECONDS)
            except requests.HTTPError as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                return None

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_el = soup.select_one("h1.entry-title, h1, .post-title")
        title = title_el.get_text(strip=True) if title_el else "Untitled"

        # Extract date
        published_date = None
        time_el = soup.select_one("time.entry-date, time")
        if time_el and time_el.get("datetime"):
            try:
                published_date = datetime.fromisoformat(
                    time_el["datetime"].replace("Z", "+00:00")
                ).date()
            except ValueError:
                pass

        if not published_date:
            # Try parsing from URL pattern: /yyyy/mm/dd/
            date_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
            if date_match:
                try:
                    published_date = datetime(
                        int(date_match.group(1)),
                        int(date_match.group(2)),
                        int(date_match.group(3)),
                    ).date()
                except ValueError:
                    pass

        if not published_date:
            logger.warning(f"Could not parse date for {url}")
            return None

        # Extract author
        author = None
        author_el = soup.select_one(".byline a, .author a, .entry-author a")
        if author_el:
            author = author_el.get_text(strip=True)

        # Extract main content
        content_el = soup.select_one(
            "div.entry-content, div.post-content, article .content"
        )
        if not content_el:
            logger.warning(f"No content found for {url}")
            return None

        # Remove unwanted elements
        for tag in content_el.select(
            "script, style, .sharedaddy, .jp-relatedposts, .entry-meta"
        ):
            tag.decompose()

        # Get text from paragraphs
        paragraphs = content_el.find_all("p")
        if paragraphs:
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        else:
            text = content_el.get_text(separator="\n", strip=True)

        if not text or len(text) < 50:
            logger.warning(f"Post text too short for {url}: {len(text)} chars")
            return None

        word_count = len(text.split())
        expedition_id = map_date_to_expedition(published_date, self.expeditions)

        return BlogPost(
            url=url,
            title=title,
            author=author,
            published_date=published_date,
            text=text,
            word_count=word_count,
            expedition_id=expedition_id,
        )

    def detect_max_pages(self) -> int:
        """Fetch page 1 and detect the total number of listing pages."""
        html = self._get_cached_html("listing_0001")
        if html is None:
            resp = fetch_page(LISTING_URL.format(page_num=1))
            html = resp.text
            self._cache_html("listing_0001", html)
            time.sleep(RATE_LIMIT_SECONDS)

        soup = BeautifulSoup(html, "html.parser")

        # Look for pagination links to find the max page number
        page_links = soup.select("a.page-numbers, .pagination a")
        max_page = 1
        for link in page_links:
            text = link.get_text(strip=True)
            if text.isdigit():
                max_page = max(max_page, int(text))

        # Also check for href patterns
        for link in page_links:
            href = link.get("href", "")
            match = re.search(r"/page/(\d+)", href)
            if match:
                max_page = max(max_page, int(match.group(1)))

        logger.info(f"Detected {max_page} listing pages")
        return max_page

    def run(self, max_pages: int | None = None, mvp_mode: bool = False) -> None:
        """Run the full blog scraping pipeline.

        Args:
            max_pages: Override maximum number of listing pages to scrape.
            mvp_mode: If True, only scrape the first 100 pages (~1K recent posts).
        """
        checkpoint = self._load_checkpoint()
        scraped_urls = set(checkpoint.get("scraped_post_urls", []))
        start_page = checkpoint.get("last_listing_page", 0) + 1

        if max_pages is None:
            if mvp_mode:
                max_pages = 100
            else:
                max_pages = self.detect_max_pages()

        logger.info(
            f"Starting blog scrape: pages {start_page}-{max_pages}, "
            f"{len(scraped_urls)} posts already scraped"
        )

        for page_num in range(start_page, max_pages + 1):
            try:
                post_metas = self.scrape_listing_page(page_num)
            except Exception as e:
                logger.error(f"Failed to scrape listing page {page_num}: {e}")
                continue

            for meta in post_metas:
                url = meta["url"]
                if url in scraped_urls:
                    continue

                try:
                    post = self.scrape_full_post(url)
                    if post:
                        row_id = insert_blog_post(self.conn, post)
                        if row_id > 0:
                            logger.info(
                                f"Inserted: {post.title[:60]} ({post.published_date})"
                            )
                        scraped_urls.add(url)
                except Exception as e:
                    logger.error(f"Failed to scrape post {url}: {e}")
                    continue

            # Update checkpoint after each listing page
            checkpoint["last_listing_page"] = page_num
            checkpoint["scraped_post_urls"] = list(scraped_urls)
            self._save_checkpoint(checkpoint)

        logger.info(f"Blog scrape complete. Total posts scraped: {len(scraped_urls)}")
