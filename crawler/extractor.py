import json
import logging
import os
from urllib.parse import urlparse
from typing import Optional
from google import genai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment.")

client = genai.Client(api_key=api_key) if api_key else None

# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Output
# ---------------------------------------------------------------------------
class SubpageSelection(BaseModel):
    urls: list[str]

class ProfessorData(BaseModel):
    name: str
    title: Optional[str] = None

class ResearchThemeData(BaseModel):
    title: str
    description: Optional[str] = None
    approach: Optional[str] = None  # Specific method / technique used

class LabExtractionResult(BaseModel):
    name: str
    name_en: Optional[str] = None
    department: Optional[str] = None
    faculty: Optional[str] = None
    description: Optional[str] = None        # Synthesized description of what makes this lab unique
    vision: Optional[str] = None             # Lab's research philosophy / mission / ultimate goal
    signature_research: list[str] = []       # Characteristic / flagship research topics specific to this lab
    keywords: list[str]
    professors: list[ProfessorData]
    themes: list[ResearchThemeData]

# ---------------------------------------------------------------------------
# Subpage Selection
# ---------------------------------------------------------------------------
async def select_subpages(links: list[dict], base_url: str) -> list[str]:
    """
    Use Gemini to select up to 2 subpages likely containing research content
    and publications/paper lists. Member and access pages are excluded.
    """
    if not client or not links:
        return []

    # Filter to same domain only, remove the base URL itself
    base_domain = urlparse(base_url).netloc
    same_domain = [
        l for l in links
        if urlparse(l.get("url", "")).netloc == base_domain
        and l.get("url", "").rstrip("/") != base_url.rstrip("/")
    ]

    # Deduplicate by URL
    seen: set[str] = set()
    unique_links: list[dict] = []
    for l in same_domain:
        url = l.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique_links.append(l)

    if not unique_links:
        return []

    links_text = "\n".join(
        f"- [{l.get('text', '').strip()}]({l['url']})"
        for l in unique_links[:100]
    )

    prompt = f"""以下は京都大学の研究室ウェブサイト（{base_url}）内のリンク一覧です。

この研究室の「研究内容・研究テーマ」および「業績・論文リスト」が書かれていると思われるページのURLを最大2件選んでください。
メンバー紹介・アクセス・お知らせ・採用情報のページは選ばないでください。
該当するページが存在しない場合は空リストを返してください。

リンク一覧:
{links_text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SubpageSelection,
                temperature=0.0,
            ),
        )
        if not response.text:
            return []
        data = json.loads(response.text)
        return data.get("urls", [])[:2]
    except Exception as e:
        logger.error(f"Error selecting subpages: {e}")
        return []

# ---------------------------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------------------------
async def extract_lab_data(text: str) -> Optional[LabExtractionResult]:
    """
    Extracts structured laboratory information from combined webpage text using Gemini API.
    The text may include content from multiple pages (main page + subpages), separated by headers.
    """
    if not client:
        logger.error("Cannot extract data: GEMINI_API_KEY is missing/client not initialized.")
        return None

    try:
        prompt = f"""You are an expert at extracting structured information from Kyoto University laboratory websites.
The text below may include content from multiple pages (main page, research page, publications page), separated by "===" headers.
The text may be in Japanese, English, or a mix of both. Extract information with equal depth regardless of the source language.

Your goal is to capture what makes this laboratory DISTINCTIVE — not generic field descriptions, but the specific problems, approaches, and vision unique to this lab.

## Fields to extract:

**name / name_en**: Official lab name in Japanese and English (extract both if present).

**department / faculty**: Affiliation within Kyoto University.

**description**:
- Write 3–5 sentences capturing what specific problems this lab tackles, how they approach them, and what distinguishes them from other labs in the same field.
- IMPORTANT: Do NOT copy or quote the introductory paragraph verbatim. Synthesize information from across all provided pages (research themes, paper titles, project descriptions, etc.) and write in your own words.
- Avoid vague statements like "conducts cutting-edge research in X". Be specific (e.g., "Develops sparse Bayesian methods for real-time decoding of neural signals to restore motor function, implemented on custom FPGA hardware").

**vision**:
- The lab's stated research philosophy, mission, or ultimate goal — WHY this lab exists and what it ultimately aims to achieve.
- Look for mission statements, introductory paragraphs, or "about" sections.
- If not explicitly stated, infer from the overall direction of their research themes and paper topics.
- Do NOT leave this null unless there is truly no way to infer it.

**signature_research**:
- 3–5 specific, characteristic research topics or projects that are unique to this lab.
- Draw from paper titles, project names, and theme descriptions — these are the most reliable sources.
- Be concrete and specific (e.g., "マルチモーダル感情認識のためのトランスフォーマー適応" or "topology-aware routing in delay-tolerant networks"), not generic field names like "machine learning" or "robotics".

**keywords**: 5–15 specific technical terms, methods, or application areas. Prefer domain-specific terms over broad fields.

**professors**: All faculty members with title (教授/准教授/講師/Professor/Associate Professor etc.) separated from name.

**themes**: Individual research themes or projects.
- title: Specific theme name
- description: What exactly is being researched and why it matters
- approach: The specific method, technique, or framework used (if mentioned)

## Important rules:
- Synthesize across all page sections — do not just extract from the first section.
- Extract at the same level of detail whether the source is Japanese or English.
- Output field values in the language of the source text. Do not translate.
- Do not fabricate information not present in the text.
- Separate professor titles from names (e.g., name="山田 太郎", title="教授").

Webpage Text:
---------------------
{text[:25000]}
---------------------
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LabExtractionResult,
                temperature=0.1,
            ),
        )

        if not response.text:
            logger.error("Gemini API returned an empty response.")
            return None

        data_dict = json.loads(response.text)
        return LabExtractionResult.model_validate(data_dict)

    except Exception as e:
        logger.error(f"Error extracting data with Gemini: {e}")
        return None
