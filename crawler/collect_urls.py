"""
crawler/collect_urls.py

crawler/index_urls.txt に記載されたインデックスページのURLから
研究室サイトのURLを抽出し、crawler/urls.csv に保存するスクリプト。

index_urls.txt の # コメントから faculty / department を読み取る:
  # ========== 京都大学大学院 工学研究科 ========== → faculty
  # 社会基盤工学専攻                               → department

使い方:
  docker run --rm --network host kyoto-u-lab-matching-crawler python crawler/collect_urls.py

URL ごとの抽出戦略:
  - t.kyoto-u.ac.jp (工学研究科): navTree → 研究室セクション → 研究室Webサイト
  - それ以外 (情報学研究科等): 「研究室サイトへ」リンク
"""

import asyncio
import csv
import logging
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright, Browser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_URLS_FILE = Path(__file__).parent / "index_urls.txt"
OUTPUT_CSV = Path(__file__).parent / "urls.csv"


def load_index_entries() -> list[dict]:
    """
    index_urls.txt を解析して (url, faculty, department) のリストを返す。

    # === ... 工学研究科 === → current_faculty を更新
    # 社会基盤工学専攻       → current_department を更新
    URL行                    → (url, current_faculty, current_department) を追加
    """
    if len(sys.argv) > 1:
        # コマンドライン引数でURLが指定された場合はそちらを使う（faculty/department は空）
        return [{"url": u, "faculty": "", "department": ""} for u in sys.argv[1:]]

    if not INDEX_URLS_FILE.exists():
        logger.error(f"{INDEX_URLS_FILE} が見つかりません。")
        sys.exit(1)

    entries: list[dict] = []
    current_faculty = ""
    current_department = ""

    with open(INDEX_URLS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                comment = line.lstrip("#").strip()
                # セクションヘッダー: "===...=== 〇〇研究科 ===...==="
                m = re.search(r'([^\s=]+研究科)', comment)
                if m:
                    current_faculty = m.group(1)
                    current_department = ""
                elif comment:
                    # 専攻・コース名: 括弧書き注記（環境）などを除去
                    dept = re.sub(r'[（(][^）)]*[）)]', '', comment).strip()
                    if dept:
                        current_department = dept
            else:
                # URL行
                entries.append({
                    "url": line,
                    "faculty": current_faculty,
                    "department": current_department,
                })

    logger.info(f"Loaded {len(entries)} index entries from {INDEX_URLS_FILE}")
    return entries


def is_engineering_url(url: str) -> bool:
    return "t.kyoto-u.ac.jp" in url


# ---------------------------------------------------------------------------
# 情報学研究科: 「研究室サイトへ」リンクを抽出
# ---------------------------------------------------------------------------

async def extract_ist_lab_urls(
    browser: Browser, index_url: str, faculty: str, department: str
) -> list[dict]:
    logger.info(f"[情報学] {index_url}")
    results = []
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


# ---------------------------------------------------------------------------
# 工学研究科: navTree → 研究室セクション → 研究室Webサイト or ページ自体
# ---------------------------------------------------------------------------

async def extract_eng_lab_urls(
    browser: Browser, index_url: str, faculty: str, department: str
) -> list[dict]:
    """
    1. index_url の navTree から専攻内グループページのリンクを取得
    2. 各グループページで <h2>研究室</h2> 以下のリンクを取得
    3. 各リンク先で「研究室Webサイト」があればそのURL、なければリンク先自体を使う
    """
    logger.info(f"[工学] {index_url}")
    results = []

    # Step 1: navTree からサブページURLを取得
    page = await browser.new_page()
    try:
        await page.goto(index_url, wait_until="networkidle", timeout=30000)
        nav_hrefs: list[str] = await page.evaluate("""
            () => {
                const ul = document.querySelector('ul.navTree.navTreeLevel1');
                if (!ul) return [];
                return [...ul.querySelectorAll('a[href]')].map(a => a.href);
            }
        """)
        logger.info(f"  navTree sub-pages: {nav_hrefs}")
        if not nav_hrefs:
            nav_hrefs = [index_url]
    except Exception as e:
        logger.error(f"Error fetching {index_url}: {e}")
        await page.close()
        return results
    finally:
        await page.close()

    # Step 2: 各サブページで <h2>研究室</h2> 以下のリンクを取得
    for nav_url in nav_hrefs:
        page = await browser.new_page()
        try:
            await page.goto(nav_url, wait_until="networkidle", timeout=30000)
            lab_links: list[dict] = await page.evaluate("""
                () => {
                    const h2s = [...document.querySelectorAll('h2')];
                    const h2 = h2s.find(el => el.innerText.trim() === '研究室');
                    if (!h2) return [];
                    const links = [];
                    let el = h2.nextElementSibling;
                    while (el && el.tagName !== 'H2') {
                        el.querySelectorAll('a[href]').forEach(a => {
                            if (a.href) links.push({ href: a.href, text: a.innerText.trim() });
                        });
                        el = el.nextElementSibling;
                    }
                    return links;
                }
            """)
            logger.info(f"  研究室 links ({nav_url}): {[l['text'] for l in lab_links]}")
        except Exception as e:
            logger.error(f"Error fetching sub-page {nav_url}: {e}")
            await page.close()
            continue
        finally:
            await page.close()

        # Step 3: 各研究室ページで「研究室Webサイト」リンクを探す
        for lab_link in lab_links:
            lab_detail_url = lab_link["href"]
            page = await browser.new_page()
            try:
                await page.goto(lab_detail_url, wait_until="networkidle", timeout=30000)
                website_href: str | None = await page.evaluate("""
                    () => {
                        const h2s = [...document.querySelectorAll('h2')];
                        const h2 = h2s.find(el => el.innerText.trim() === '研究室ウェブサイト');
                        if (!h2) return null;
                        let el = h2.nextElementSibling;
                        while (el && el.tagName !== 'H2') {
                            const a = el.querySelector('a[href]');
                            if (a) return a.href;
                            el = el.nextElementSibling;
                        }
                        return null;
                    }
                """)
                final_url = website_href if website_href else lab_detail_url
                logger.info(
                    f"  → {'Webサイト' if website_href else 'ページ自体'}: {final_url}"
                )
                results.append({
                    "url": final_url,
                    "source_page": index_url,
                    "faculty": faculty,
                    "department": department,
                })
            except Exception as e:
                logger.error(f"Error visiting lab detail {lab_detail_url}: {e}")
            finally:
                await page.close()

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main():
    entries = load_index_entries()

    all_labs: list[dict] = []
    seen_urls: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            for entry in entries:
                url = entry["url"]
                faculty = entry["faculty"]
                department = entry["department"]

                if is_engineering_url(url):
                    labs = await extract_eng_lab_urls(browser, url, faculty, department)
                else:
                    labs = await extract_ist_lab_urls(browser, url, faculty, department)

                for lab in labs:
                    if lab["url"] not in seen_urls:
                        seen_urls.add(lab["url"])
                        all_labs.append(lab)
        finally:
            await browser.close()

    if not all_labs:
        logger.warning("No lab URLs found. Check the index page structure.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "source_page", "faculty", "department"])
        writer.writeheader()
        writer.writerows(all_labs)

    logger.info(f"\n✅ {len(all_labs)} lab URLs saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
