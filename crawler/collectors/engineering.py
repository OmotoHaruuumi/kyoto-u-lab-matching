"""
工学研究科タイプの収集ストラテジ。

対象サイト構造（t.kyoto-u.ac.jp 系）:
  1. 専攻ページの navTree（ul.navTree.navTreeLevel1）から専攻内グループページのリンクを取得
  2. 各グループページで <h2>研究室</h2> 以下のリンク（研究室詳細ページ）を取得
  3. 各研究室詳細ページで <h2>研究室ウェブサイト</h2> 以下のリンクを取得
     （無ければ研究室詳細ページ自体のURLを使う）

工学と同じCMS構造の研究科があれば matches() を広げて流用できる。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Browser

logger = logging.getLogger(__name__)

NAME = "engineering"


def matches(index_url: str) -> bool:
    return "t.kyoto-u.ac.jp" in index_url


async def collect(
    browser: Browser, index_url: str, faculty: str, department: str
) -> list[dict]:
    results: list[dict] = []
    logger.info(f"[{NAME}] {index_url}")

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

        # Step 3: 各研究室ページで「研究室ウェブサイト」リンクを探す
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
