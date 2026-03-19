import asyncio
import logging
import sys

from playwright.async_api import async_playwright

# Fix path to import backend modules
sys.path.append(".")

from backend.shared.database import async_session_maker
from crawler.extractor import extract_lab_data
from crawler.loader import store_lab_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_page_content(url: str) -> str:
    """Fetch text content of a webpage using Playwright."""
    logger.info(f"Navigating to {url}...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            # Wait until network is idle or 30s timeout
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Extract plain text from page using innerText
            text_content = await page.evaluate("document.body.innerText")
            return text_content or ""
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
        finally:
            await browser.close()

async def crawl_lab_webpage(url: str):
    """
    Main flow for crawling a single lab webpage:
    1. Fetch raw text via Playwright
    2. Extract structured data via Gemini
    3. Save to database via loader
    """
    logger.info(f"Starting crawl for {url}")
    
    # 1. Fetch
    raw_text = await fetch_page_content(url)
    if not raw_text:
        logger.warning(f"No text content found at {url}. Skipping.")
        return

    logger.info(f"Extracted {len(raw_text)} characters from {url}. Passing to Gemini...")
    
    # 2. Extract
    extracted_data = await extract_lab_data(raw_text)
    if not extracted_data:
        logger.error(f"Failed to extract structured data for {url}. Skipping.")
        return
        
    logger.info(f"Successfully extracted data for lab: {extracted_data.name}. Saving to DB...")

    # 3. Load
    async with async_session_maker() as session:
        try:
             success = await store_lab_data(session, url, raw_text, extracted_data)
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
        # フォールバック: CSVがない場合のデフォルトURL
        logger.info("Using fallback default URLs.")
        target_urls = [
            "https://nlp.ist.i.kyoto-u.ac.jp/",
        ]

    for url in target_urls:
        await crawl_lab_webpage(url)

if __name__ == "__main__":
    asyncio.run(main())
