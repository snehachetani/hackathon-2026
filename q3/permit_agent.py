import os, json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

client = genai.Client(api_key=os.getenv("gemini_api_key"))
MODEL = "gemini-2.5-flash-lite"

PROMPT = """You are a German work permit (Aufenthaltstitel / Arbeitserlaubnis) document validator.

Analyze the document text and return ONLY a JSON object with exactly these fields:
{
  "is_work_permit": true or false,
  "confidence": integer 0-100 (how confident you are this is a valid work permit),
  "document_type": "exact permit type or null",
  "holder_name": "full name or null",
  "valid_until": "DD.MM.YYYY or null",
  "is_currently_valid": true or false or null,
  "employment_type": "exact employment permission text from the document, e.g. 'Dependent employment permitted' or 'Any employment including self-employment permitted' or null",
  "issuing_authority": "authority name or null",
  "reason": "one sentence explaining your confidence score"
}

For is_currently_valid: compare valid_until against today's date 2026-06-20. If expired → false, if still future → true.
For confidence: 85-100 if all key fields present and structure matches a genuine German Aufenthaltstitel,
50-84 if some fields missing, 0-49 if not a work permit at all.

Return only valid JSON, no markdown."""


def validate_permit(path: Path) -> dict:
    reader = PdfReader(str(path))
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    response = client.models.generate_content(model=MODEL, contents=[PROMPT, f"Document text:\n{text}"])
    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
