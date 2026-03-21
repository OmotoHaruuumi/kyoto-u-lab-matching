import asyncio
import logging
import sys

from playwright.async_api import async_playwright, Browser

# Fix path to import backend modules
sys.path.append(".")

from backend.shared.database import async_session_maker
from crawler.extractor import extract_lab_data, select_subpages
from crawler.loader import check_url_crawled, store_lab_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Per-page character limits when combining texts
MAIN_PAGE_CHAR_LIMIT = 10000
SUBPAGE_CHAR_LIMIT = 7500


async def fetch_page(browser: Browser, url: str) -> str:
    """Fetch text content of a single page using an existing browser instance."""
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        return await page.evaluate("document.body.innerText") or ""
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return ""
    finally:
        await page.close()


async def fetch_page_with_links(browser: Browser, url: str) -> tuple[str, list[dict]]:
    """Fetch page text and extract all internal links with their anchor text."""
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        text = await page.evaluate("document.body.innerText") or ""
        links = await page.evaluate(
            """() => [...document.querySelectorAll('a[href]')]
                .map(a => ({url: a.href, text: a.innerText.trim()}))
                .filter(l => l.text.length > 0 && l.url.startsWith('http'))"""
        )
        return text, links
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return "", []
    finally:
        await page.close()


def combine_page_texts(main_url: str, main_text: str, subpages: list[tuple[str, str]]) -> str:
    """Combine main page and subpage texts with section headers and per-page character limits."""
    parts = [f"=== メインページ: {main_url} ===\n{main_text[:MAIN_PAGE_CHAR_LIMIT]}"]
    for url, text in subpages:
        if text:
            parts.append(f"=== サブページ: {url} ===\n{text[:SUBPAGE_CHAR_LIMIT]}")
    return "\n\n".join(parts)


async def crawl_lab_webpage(url: str):
    """
    Main flow for crawling a single lab webpage:
    1. Fetch main page + extract internal links
    2. Use Gemini to select subpages (research content / publications)
    3. Fetch selected subpages
    4. Combine all text and extract structured data via Gemini
    5. Save to database
    """
    logger.info(f"Starting crawl for {url}")

    async with async_session_maker() as session:
        if await check_url_crawled(session, url):
            logger.info(f"Already crawled, skipping: {url}")
            return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            # 1. Fetch main page and extract links
            main_text, links = await fetch_page_with_links(browser, url)
            if not main_text:
                logger.warning(f"No text content found at {url}. Skipping.")
                return
            logger.info(f"Main page: {len(main_text)} chars, {len(links)} links found.")

            # 2. Select relevant subpages via Gemini
            subpage_urls = await select_subpages(links, base_url=url)
            logger.info(f"Selected subpages: {subpage_urls}")

            # 3. Fetch subpages
            subpage_texts: list[tuple[str, str]] = []
            for sub_url in subpage_urls:
                text = await fetch_page(browser, sub_url)
                subpage_texts.append((sub_url, text))
                logger.info(f"Subpage {sub_url}: {len(text)} chars.")

        finally:
            await browser.close()

    # 4. Combine texts and extract
    combined_text = combine_page_texts(url, main_text, subpage_texts)
    logger.info(f"Combined text: {len(combined_text)} chars total. Passing to Gemini...")

    extracted_data = await extract_lab_data(combined_text)
    if not extracted_data:
        logger.error(f"Failed to extract structured data for {url}. Skipping.")
        return
    logger.info(f"Extracted data for: {extracted_data.name}")

    # 5. Store
    async with async_session_maker() as session:
        try:
            success = await store_lab_data(session, url, combined_text, extracted_data)
            if success:
                logger.info(f"Pipeline complete for {url}.")
            else:
                logger.error(f"Pipeline failed at storage step for {url}.")
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error while saving data for {url}: {e}", exc_info=True)


def load_urls_from_csv(csv_path: str) -> list[str]:
    """CSVファイルからURLリストを読み込む。"""
    import csv as csv_module
    urls = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                url = row.get("url", "").strip()
                if url:
                    urls.append(url)
        logger.info(f"Loaded {len(urls)} URLs from {csv_path}")
    except FileNotFoundError:
        logger.warning(f"CSV not found: {csv_path}. Using default URLs.")
    return urls


async def main():
    """CSVファイルからURLを読み込んでクロールする。CSVがなければデフォルトURLを使用。"""
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "urls.csv")
    target_urls = load_urls_from_csv(csv_path)

    if not target_urls:
        logger.info("Using fallback default URLs.")
        target_urls = [
            "https://nlp.ist.i.kyoto-u.ac.jp/",
        ]

    for url in target_urls:
        await crawl_lab_webpage(url)


if __name__ == "__main__":
    asyncio.run(main())
