"""
crawler/collect_urls.py

crawler/index_urls.txt に記載されたインデックスページのURLから研究室サイトのURLを抽出し、
crawler/urls.csv（4列: url, source_page, faculty, department）に保存する。

収集ロジック本体は collectors/ パッケージに分割されている。このスクリプトは
「どの collector を使うかの振り分け」と「研究科フィルタ」「CSVへのマージ/上書き」だけを担う。

index_urls.txt の # コメントから faculty / department を読み取る:
  # ========== 京都大学大学院 工学研究科 ========== → faculty
  # 社会基盤工学専攻                               → department

使い方:
  # 全研究科を収集して urls.csv を全面上書き
  python crawler/collect_urls.py

  # 特定の研究科だけ収集して、その研究科ぶんだけ urls.csv にマージ（他研究科は保持）
  python crawler/collect_urls.py --faculty 工学研究科

  # 任意のURLを直接指定して収集（faculty/department は空でマージ）
  python crawler/collect_urls.py --url https://example.kyoto-u.ac.jp/

  Docker:
  docker run --rm --network host kyoto-u-lab-matching-crawler \\
      python crawler/collect_urls.py --faculty 工学研究科
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import sys
from pathlib import Path

# crawler パッケージを import 可能にする（リポジトリルートを sys.path に追加）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.collectors import select_collector  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_URLS_FILE = Path(__file__).parent / "index_urls.txt"
OUTPUT_CSV = Path(__file__).parent / "urls.csv"
FIELDNAMES = ["url", "source_page", "faculty", "department"]


def load_index_entries(faculty_filter: str | None = None) -> list[dict]:
    """
    index_urls.txt を解析して (url, faculty, department) のリストを返す。

    # === ... 工学研究科 === → current_faculty を更新
    # 社会基盤工学専攻       → current_department を更新
    URL行                    → (url, current_faculty, current_department) を追加

    faculty_filter を指定すると、その研究科のエントリだけを返す。
    """
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
                if faculty_filter and current_faculty != faculty_filter:
                    continue
                entries.append({
                    "url": line,
                    "faculty": current_faculty,
                    "department": current_department,
                })

    if faculty_filter:
        logger.info(f"Loaded {len(entries)} index entries for {faculty_filter!r}")
    else:
        logger.info(f"Loaded {len(entries)} index entries (all faculties)")
    return entries


async def collect_entries(entries: list[dict]) -> list[dict]:
    """各インデックスエントリを担当 collector に振り分けて研究室URLを収集する。"""
    from playwright.async_api import async_playwright  # 実行時にのみ必要

    all_labs: list[dict] = []
    seen_urls: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            for entry in entries:
                collector = select_collector(entry["url"])
                labs = await collector.collect(
                    browser, entry["url"], entry["faculty"], entry["department"]
                )
                for lab in labs:
                    if lab["url"] not in seen_urls:
                        seen_urls.add(lab["url"])
                        all_labs.append(lab)
        finally:
            await browser.close()

    return all_labs


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {k: (row.get(k) or "") for k in FIELDNAMES}
            for row in csv.DictReader(f)
            if (row.get("url") or "").strip()
        ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_results(new_rows: list[dict], faculty_filter: str | None) -> None:
    """
    faculty_filter 指定あり → 既存 urls.csv をマージ（対象研究科ぶんを差し替え、他は保持）。
    指定なし               → 全面上書き。
    """
    if not faculty_filter:
        _write_csv(OUTPUT_CSV, new_rows)
        logger.info(f"\n✅ {len(new_rows)} lab URLs saved to {OUTPUT_CSV}（全面上書き）")
        return

    existing = _read_existing(OUTPUT_CSV)
    new_urls = {r["url"] for r in new_rows}
    # 対象研究科の古い行 と URL重複行 を取り除き、他研究科は保持
    kept = [
        r for r in existing
        if r["faculty"] != faculty_filter and r["url"] not in new_urls
    ]
    merged = kept + new_rows
    _write_csv(OUTPUT_CSV, merged)
    logger.info(
        f"\n✅ {faculty_filter}: {len(new_rows)}件をマージ "
        f"（保持 {len(kept)}件 / 合計 {len(merged)}件）→ {OUTPUT_CSV}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="研究室URLを収集して urls.csv に保存する")
    parser.add_argument(
        "--faculty",
        help="この研究科のインデックスだけを収集し、urls.csv にマージする（他研究科は保持）",
    )
    parser.add_argument(
        "--url",
        nargs="+",
        help="任意のURLを直接指定して収集（faculty/department は空でマージ）",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.url:
        entries = [{"url": u, "faculty": "", "department": ""} for u in args.url]
        faculty_filter = None  # 任意URLは全面上書きせずマージ扱いにする
        labs = await collect_entries(entries)
        # 任意URL指定時は既存を保持してマージ（faculty 空のものを差し替え）
        existing = _read_existing(OUTPUT_CSV)
        new_urls = {r["url"] for r in labs}
        kept = [r for r in existing if r["url"] not in new_urls]
        _write_csv(OUTPUT_CSV, kept + labs)
        logger.info(f"\n✅ {len(labs)}件をマージ（保持 {len(kept)}件）→ {OUTPUT_CSV}")
        return

    faculty_filter = args.faculty
    entries = load_index_entries(faculty_filter=faculty_filter)
    if not entries:
        logger.warning("対象のインデックスエントリがありません。index_urls.txt を確認してください。")
        return

    labs = await collect_entries(entries)
    if not labs:
        logger.warning("No lab URLs found. Check the index page structure / collector.")
        return

    save_results(labs, faculty_filter)


if __name__ == "__main__":
    asyncio.run(main())
