"""
crawler/normalize_categories.py

categories.json の内容に基づいて DB の labs.faculty / labs.department を正規化します。
一致しないレコードは変更せずに一覧表示するので、手動で categories.json を更新するか
直接 DB を編集してください。

Usage:
    python crawler/normalize_categories.py [--dry-run]
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select, update
from backend.shared.database import async_session_maker
from backend.shared.models import Lab

CATEGORIES_PATH = Path(__file__).parent / "categories.json"


async def main(dry_run: bool = False) -> None:
    categories: dict[str, list[str]] = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))

    # Build lookup sets (小文字で比較)
    faculty_set = {f.lower(): f for f in categories}
    dept_lookup: dict[str, tuple[str, str]] = {}  # dept_lower -> (faculty, dept)
    for faculty, depts in categories.items():
        for dept in depts:
            dept_lookup[dept.lower()] = (faculty, dept)

    async with async_session_maker() as session:
        result = await session.execute(select(Lab))
        labs = result.scalars().all()

        matched = []
        unmatched = []

        for lab in labs:
            new_faculty = None
            new_dept = None

            # faculty のマッチ
            if lab.faculty:
                for key, canonical in faculty_set.items():
                    if key in (lab.faculty or "").lower():
                        new_faculty = canonical
                        break

            # department のマッチ
            if lab.department:
                for key, (fac, dept) in dept_lookup.items():
                    if key in (lab.department or "").lower():
                        new_dept = dept
                        if new_faculty is None:
                            new_faculty = fac
                        break

            if new_faculty or new_dept:
                matched.append((lab, new_faculty, new_dept))
            else:
                unmatched.append(lab)

        print(f"\n=== マッチ: {len(matched)} 件 ===")
        for lab, fac, dept in matched:
            print(f"  [{lab.id}] {lab.name}")
            print(f"    faculty:    {lab.faculty!r} -> {fac!r}")
            print(f"    department: {lab.department!r} -> {dept!r}")

        print(f"\n=== 未マッチ（categories.json にないため null に設定）: {len(unmatched)} 件 ===")
        for lab in unmatched:
            print(f"  [{lab.id}] {lab.name} | faculty={lab.faculty!r} -> None | department={lab.department!r} -> None")

        if dry_run:
            print("\n[dry-run] DB は更新しませんでした。")
            return

        for lab, fac, dept in matched:
            if fac:
                lab.faculty = fac
            if dept:
                lab.department = dept

        for lab in unmatched:
            lab.faculty = None
            lab.department = None

        await session.commit()
        print(f"\nマッチ: {len(matched)} 件を正規化、未マッチ: {len(unmatched)} 件を null に設定しました。")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry_run))
