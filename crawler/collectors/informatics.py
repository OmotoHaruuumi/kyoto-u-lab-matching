"""
情報学研究科タイプの収集ストラテジ。

対象サイト構造: コース一覧ページに「研究室サイトへ」というリンクが並んでおり、
そのリンク先が各研究室のWebサイト。
（情報学研究科 i.kyoto-u.ac.jp 系）

新しい研究科でも同じ「研究室サイトへ」リンク方式なら matches() を広げて流用できる。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Browser

logger = logging.getLogger(__name__)

NAME = "informatics"


def matches(index_url: str) -> bool:
    return "i.kyoto-u.ac.jp" in index_url


async def collect(
    browser: Browser, index_url: str, faculty: str, department: str
) -> list[dict]:
    """インデックスページから「研究室サイトへ」リンクを抽出する。"""
    logger.info(f"[{NAME}] {index_url}")
    results: list[dict] = []
    page = await browser.new_page()
    try:
        await page.goto(index_url, wait_until="networkidle", timeout=30000)
        links = await page.query_selector_all("a")
        for link in links:
            text = (await link.inner_text()).strip()
            href = await link.get_attribute("href")
            if href and "研究室サイトへ" in text:
                results.append({
                    "url": href,
                    "source_page": index_url,
                    "faculty": faculty,
                    "department": department,
                })
                logger.info(f"  Found: {href}")
    except Exception as e:
        logger.error(f"Error fetching {index_url}: {e}")
    finally:
        await page.close()
    return results
