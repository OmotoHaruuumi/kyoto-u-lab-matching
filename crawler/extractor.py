import json
import logging
import os
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
class ProfessorData(BaseModel):
    name: str
    title: Optional[str] = None

class ResearchThemeData(BaseModel):
    title: str
    description: Optional[str] = None

class LabExtractionResult(BaseModel):
    name: str
    name_en: Optional[str] = None
    department: Optional[str] = None
    faculty: Optional[str] = None
    description: Optional[str] = None
    keywords: list[str]
    professors: list[ProfessorData]
    themes: list[ResearchThemeData]

# ---------------------------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------------------------
async def extract_lab_data(text: str) -> Optional[LabExtractionResult]:
    """
    Extracts structured laboratory information from raw webpage text using Gemini API.
    """
    if not client:
        logger.error("Cannot extract data: GEMINI_API_KEY is missing/client not initialized.")
        return None

    try:
        prompt = f"""
        You are an expert at extracting structured information from academic laboratory websites. Your task is to extract information about Kyoto University laboratories accurately.
        
        Extract the laboratory's information from the following webpage text. 
        Focus on finding the lab's name, department/faculty affiliations, a general description, a list of keywords, faculty members (professors, associate professors, etc.), and their specific research themes.
        If a piece of information is missing, leave it as null or omit it according to the schema.
        Ensure names are clean and titles are separated from names if possible.

        Webpage Text:
        ---------------------
        {text[:25000]} # Limit text length to avoid token limits just in case
        ---------------------
        """

        # Call the new iterative genai Client API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
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

        # Parse the JSON string into the Pydantic model
        data_dict = json.loads(response.text)
        return LabExtractionResult.model_validate(data_dict)

    except Exception as e:
        logger.error(f"Error extracting data with Gemini: {e}")
        return None
