"""
db/seed.py

Insert 5 dummy Kyoto University labs, their professors, and research themes.
Fetches embeddings by making a POST request to http://localhost:8001/api/v1/embed
and stores the retrieved vectors into embedding_chunks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

import httpx
from sqlalchemy import select

# Fix path to import backend modules
sys.path.append(".")

# Override DATABASE_URL for local execution (outside docker)
os.environ["DATABASE_URL"] = "postgresql+asyncpg://labmatch:labmatch_secret@localhost:5432/labmatch_db"

from backend.shared.database import async_session_maker
from backend.shared.models import EmbeddingChunk, Lab, Professor, ResearchTheme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_API_URL = "http://localhost:8001/api/v1/embed"


# ---------------------------------------------------------------------------
# Dummy Data
# ---------------------------------------------------------------------------
DUMMY_LABS = [
    {
        "name": "自然言語処理研究室",
        "name_en": "Natural Language Processing Laboratory",
        "department": "情報学研究科",
        "faculty": "知能情報学専攻",
        "lab_url": "https://nlp.example.kyoto-u.ac.jp",
        "description": "人間の言語をコンピュータで処理するための基礎理論と応用技術を探求します。大規模言語モデルの実用化や機械翻訳の精度向上を目指しています。",
        "keywords": ["自然言語処理", "大規模言語モデル", "機械翻訳", "AI"],
        "professors": [
            {"name": "言葉 太郎", "title": "教授", "email": "kotoba@example.com"},
            {"name": "翻訳 花子", "title": "准教授", "email": "honyaku@example.com"},
        ],
        "themes": [
            {"title": "文脈を解釈する次世代LLMの開発", "description": "より人間に近い推論能力を持つ言語モデルを構築する。"},
            {"title": "多言語対応のリアルタイム翻訳システム", "description": "低遅延で高精度な音声・テキスト翻訳インフラの研究。"},
        ],
    },
    {
        "name": "知能ロボティクス研究室",
        "name_en": "Intelligent Robotics Lab",
        "department": "工学研究科",
        "faculty": "機械理工学専攻",
        "lab_url": "https://robotics.example.kyoto-u.ac.jp",
        "description": "人間と協調して動作する自律型ロボットの研究。深層強化学習を用いたロボット制御や、未知環境でのナビゲーション技術を開発しています。",
        "keywords": ["ロボット工学", "強化学習", "自律移動", "コンピュータビジョン"],
        "professors": [
            {"name": "機巧 次郎", "title": "教授", "email": "kikou@example.com"},
        ],
        "themes": [
            {"title": "不整地での自律歩行制御", "description": "強化学習により未知の地形でも安定して歩行できるアルゴリズム。"},
            {"title": "人間との物理的な協働作業", "description": "触覚フィードバックを活用した安全なロボットアーム制御。"},
        ],
    },
    {
        "name": "応用数学・非線形力学研究室",
        "name_en": "Applied Math & Nonlinear Dynamics",
        "department": "理学研究科",
        "faculty": "数学・数理解析専攻",
        "lab_url": "https://math.example.kyoto-u.ac.jp",
        "description": "カオス理論や非線形力学系などの純粋数学と、気象予測や生体システムへの応用数学の架け橋となる研究を行っています。",
        "keywords": ["応用数学", "非線形力学", "カオス理論", "数理モデリング"],
        "professors": [
            {"name": "数理 健一", "title": "教授", "email": "suuri@example.com"},
            {"name": "力学 恵美", "title": "助教", "email": "rikigaku@example.com"},
        ],
        "themes": [
            {"title": "生体リズムの数理モデル", "description": "心拍や睡眠サイクルを非線形振動子としてモデル化する。"},
            {"title": "気象データに潜むカオス的振る舞いの解析", "description": "長期天気予報の精度限界を力学系の観点から解明。"},
        ],
    },
    {
        "name": "グリーンエネルギー工学研究室",
        "name_en": "Green Energy Engineering Lab",
        "department": "エネルギー科学研究科",
        "faculty": "エネルギー変換科学専攻",
        "lab_url": "https://energy.example.kyoto-u.ac.jp",
        "description": "持続可能な社会に向けて、太陽光発電の高効率化や次世代蓄電池（全固体電池）の素材開発に取り組んでいます。",
        "keywords": ["再生可能エネルギー", "全固体電池", "太陽光発電", "持続可能性"],
        "professors": [
            {"name": "環境 守", "title": "教授", "email": "kankyo@example.com"},
        ],
        "themes": [
            {"title": "リチウムイオン伝導体の新素材探索", "description": "AIを用いたマテリアルズ・インフォマティクスによる新素材発見。"},
            {"title": "ペロブスカイト太陽電池の耐久性向上", "description": "熱劣化を防ぐ新しい封止技術の開発。"},
        ],
    },
    {
        "name": "ヒューマンコンピューターインタラクション研究室",
        "name_en": "HCI Laboratory",
        "department": "情報学研究科",
        "faculty": "社会情報学専攻",
        "lab_url": "https://hci.example.kyoto-u.ac.jp",
        "description": "人とコンピュータの境界をなくすUI/UXの設計。AR/VRを用いた没入型インターフェースや、視線追跡による意思決定支援システムを研究しています。",
        "keywords": ["HCI", "UI/UX", "AR/VR", "アクセシビリティ"],
        "professors": [
            {"name": "相互 結衣", "title": "教授", "email": "sogo@example.com"},
        ],
        "themes": [
            {"title": "視覚障がい者向け空間認識サポートAR", "description": "スマートフォンカメラと立体音響を用いたナビゲーション。"},
            {"title": "VR空間での多人数共同作業プラットフォーム", "description": "リモートワークにおける非言語コミュニケーションの再現。"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call the localhost embedding API to get vectors."""
    if not texts:
        return []

    async with httpx.AsyncClient() as client:
        response = await client.post(
            EMBEDDING_API_URL,
            json={"texts": texts, "task_type": "RETRIEVAL_DOCUMENT"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]


async def seed_database() -> None:
    async with async_session_maker() as session:
        # Check if already seeded
        result = await session.execute(select(Lab).limit(1))
        if result.scalars().first():
            logger.info("Database already contains labs. Skipping seed.")
            return

        logger.info("Starting database seed process...")

        for lab_data in DUMMY_LABS:
            # 1. Create Lab
            lab = Lab(
                name=lab_data["name"],
                name_en=lab_data["name_en"],
                department=lab_data["department"],
                faculty=lab_data["faculty"],
                lab_url=lab_data["lab_url"],
                description=lab_data["description"],
                keywords=lab_data["keywords"],
            )
            session.add(lab)
            await session.flush()  # To get lab.id

            # 2. Create Professors
            for prof_data in lab_data["professors"]:
                prof = Professor(
                    lab_id=lab.id,
                    name=prof_data["name"],
                    title=prof_data["title"],
                    email=prof_data["email"],
                )
                session.add(prof)

            # 3. Create Research Themes
            for theme_data in lab_data["themes"]:
                theme = ResearchTheme(
                    lab_id=lab.id,
                    title=theme_data["title"],
                    description=theme_data["description"],
                )
                session.add(theme)

            # 4. Generate Embeddings (for lab description + themes)
            chunks_to_embed = [
                {
                    "text": f"【研究室紹介】\n{lab.name} ({lab.department} {lab.faculty})\n{lab.description}\nキーワード: {', '.join(lab.keywords)}",
                    "source": "lab_description",
                }
            ]
            for theme_data in lab_data["themes"]:
                chunks_to_embed.append(
                    {
                        "text": f"【研究テーマ】\n{theme_data['title']}\n{theme_data['description']}",
                        "source": "research_theme",
                    }
                )

            texts = [c["text"] for c in chunks_to_embed]
            logger.info(f"Generating vectors for: {lab.name} ({len(texts)} chunks)")
            vectors = await get_embeddings(texts)

            # 5. Store EmbeddingChunks
            for chunk, vector in zip(chunks_to_embed, vectors):
                emb_chunk = EmbeddingChunk(
                    lab_id=lab.id,
                    chunk_text=chunk["text"],
                    source_type=chunk["source"],
                    embedding=vector,
                )
                session.add(emb_chunk)

        await session.commit()
        logger.info("Database seeding completed successfully.")


if __name__ == "__main__":
    asyncio.run(seed_database())
