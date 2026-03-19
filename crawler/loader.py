import asyncio
import logging
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import EmbeddingChunk, Lab, Professor, ResearchTheme, DataSource
from crawler.extractor import LabExtractionResult

logger = logging.getLogger(__name__)

EMBEDDING_API_URL = "http://embedding_api:8001/api/v1/embed"

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call the localhost embedding API to get vectors."""
    if not texts:
        return []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                EMBEDDING_API_URL,
                json={"texts": texts, "task_type": "RETRIEVAL_DOCUMENT"},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        # Return empty embeddings as fallback if API fails
        return [[0.0] * 768 for _ in texts]

async def check_url_crawled(session: AsyncSession, url: str) -> Optional[DataSource]:
    """Check if the URL has already been fully crawled to ensure idempotency."""
    result = await session.execute(select(DataSource).where(DataSource.url == url))
    return result.scalars().first()

async def create_data_source(session: AsyncSession, url: str) -> DataSource:
    """Create a pending crawl record. Note: lab_id will be updated later."""
    # We must have a lab_id per schema, so we create a temporary dummy lab or just wait
    # Actually, DataSource requires lab_id. We'll create it after creating the lab.
    pass

async def store_lab_data(session: AsyncSession, url: str, raw_text: str, data: LabExtractionResult) -> bool:
    """
    Store the extracted lab data into the DB, update data_source state,
    and generate vectors for search.
    """
    # 1. Check idempotency: See if this exact URL is already in successful DataSource
    # First see if any lab has this URL and was successfully crawled
    existing_ds_result = await session.execute(
        select(DataSource).where(DataSource.url == url, DataSource.status == "done")
    )
    if existing_ds_result.scalars().first():
        logger.info(f"URL already crawled successfully. Skipping: {url}")
        return True

    # Check if a Lab with the EXACT same URL already exists globally (to update instead of duplicate)
    existing_lab_result = await session.execute(
        select(Lab).where(Lab.lab_url == url)
    )
    existing_lab = existing_lab_result.scalars().first()
    
    if existing_lab:
        logger.info(f"Lab with URL {url} already exists. Currently we skip updates for brevity.")
        # If we wanted to update, we'd delete old data or update relationships here
        return True

    logger.info(f"Saving lab data for: {data.name}")

    # 2. Create Lab
    lab = Lab(
        name=data.name,
        name_en=data.name_en,
        department=data.department,
        faculty=data.faculty,
        lab_url=url,
        description=data.description,
        keywords=data.keywords,
    )
    session.add(lab)
    await session.flush() # Ensure we have lab.id

    # 3. Create Professors
    for prof_data in data.professors:
        prof = Professor(
            lab_id=lab.id,
            name=prof_data.name,
            title=prof_data.title,
        )
        session.add(prof)

    # 4. Create Research Themes
    for theme_data in data.themes:
        theme = ResearchTheme(
            lab_id=lab.id,
            title=theme_data.title,
            description=theme_data.description,
        )
        session.add(theme)

    # 5. Create Data Source record to track crawl success
    ds = DataSource(
        lab_id=lab.id,
        url=url,
        status="done"
    )
    session.add(ds)

    # 6. Generate Embeddings and create EmissionChunks
    chunks_to_embed = []
    
    # Lab description chunk
    desc_text = f"【研究室紹介】\n{lab.name} ({lab.department or ''} {lab.faculty or ''})\n{lab.description or ''}\nキーワード: {', '.join(lab.keywords or [])}"
    chunks_to_embed.append({"text": desc_text, "source": "lab_description"})

    # Professor chunks
    for p in data.professors:
         p_text = f"【教員情報】\n{p.title or ''} {p.name}\n所属: {lab.name}"
         chunks_to_embed.append({"text": p_text, "source": "professor_profile"})

    # Research Theme chunks
    for t in data.themes:
         t_text = f"【研究テーマ】\n{t.title}\n{t.description or ''}"
         chunks_to_embed.append({"text": t_text, "source": "research_theme"})

    texts = [c["text"] for c in chunks_to_embed]
    vectors = await get_embeddings(texts)

    # Save to EmbeddingChunk
    for i, (chunk, vector) in enumerate(zip(chunks_to_embed, vectors)):
        emb_chunk = EmbeddingChunk(
            lab_id=lab.id,
            chunk_text=chunk["text"],
            source_type=chunk["source"],
            embedding=vector,
        )
        session.add(emb_chunk)

    await session.commit()
    logger.info(f"Successfully committed lab data and {len(vectors)} vectors for {lab.name}.")
    return True
