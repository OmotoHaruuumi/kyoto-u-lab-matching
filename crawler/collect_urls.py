"""
crawler/collect_urls.py

crawler/index_urls.txt に記載されたインデックスページのURLから
「研究室サイトへ」のリンクを抽出し、crawler/urls.csv に保存するスクリプト。

使い方:
  1. crawler/index_urls.txt にインデックスページURLを1行1つ追記する
  2. docker run --rm --network host kyoto-u-lab-matching-crawler python crawler/collect_urls.py
  
コマンドライン引数でURLを直接指定することも可能:
  python crawler/collect_urls.py https://... https://...
"""

import asyncio
import csv
import logging
import sys
from pathlib import Path

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# index_urls.txt のパス（このスクリプトと同じディレクトリ）
INDEX_URLS_FILE = Path(__file__).parent / "index_urls.txt"

# 出力CSVのパス
OUTPUT_CSV = Path(__file__).parent / "urls.csv"


def load_index_urls() -> list[str]:
    """
    crawler/index_urls.txt からインデックスURLを読み込む。
    コマンドライン引数が指定された場合はそちらを優先する。
    """
    # コマンドライン引数が指定されていればそちらを使う
    if len(sys.argv) > 1:
        urls = sys.argv[1:]
        logger.info(f"Using {len(urls)} URLs from command-line arguments.")
        return urls

    # index_urls.txt から読み込む
    if not INDEX_URLS_FILE.exists():
        logger.error(
            f"{INDEX_URLS_FILE} が見つかりません。\n"
            "crawler/index_urls.txt を作成してインデックスページURLを1行1つ記載してください。"
        )
        sys.exit(1)

    urls = []
    with open(INDEX_URLS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        logger.error(f"{INDEX_URLS_FILE} に有効なURLが見つかりません。")
        sys.exit(1)

    logger.info(f"Loaded {len(urls)} index URLs from {INDEX_URLS_FILE}")
    return urls


async def extract_lab_urls(index_url: str) -> list[dict]:
    """
    指定したインデックスページから 「研究室サイトへ」 のリンクを抽出する。
    ラベル（分野名）と対応するURLをセットで返す。
    """
    logger.info(f"Fetching index page: {index_url}")
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(index_url, wait_until="networkidle", timeout=30000)

            # 「研究室サイトへ」リンクをすべて取得
            links = await page.query_selector_all("a")
            for link in links:
                text = (await link.inner_text()).strip()
                href = await link.get_attribute("href")
                if href and "研究室サイトへ" in text:
                    # 直前の見出し（分野名）を取得しようとトライ
                    # 見出し取得が難しいので、URLから推測する
                    results.append({
                        "url": href,
                        "source_page": index_url,
                    })
                    logger.info(f"  Found: {href}")

        except Exception as e:
            logger.error(f"Error fetching {index_url}: {e}")
        finally:
            await browser.close()

    return results


async def main():
    index_urls = load_index_urls()

    all_labs = []
    seen_urls = set()

    for index_url in index_urls:
        labs = await extract_lab_urls(index_url)
        for lab in labs:
            if lab["url"] not in seen_urls:
                seen_urls.add(lab["url"])
                all_labs.append(lab)

    if not all_labs:
        logger.warning("No lab URLs found. Check the index page structure.")
        return

    # CSVに保存
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "source_page"])
        writer.writeheader()
        writer.writerows(all_labs)

    logger.info(f"\n✅ {len(all_labs)} lab URLs saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
