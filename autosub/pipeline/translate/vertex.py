import logging
import json
from pydantic import BaseModel
from google import genai
from google.genai import types

from autosub.pipeline.translate.base import BaseTranslator

logger = logging.getLogger(__name__)


class TranslatedSubtitle(BaseModel):
    id: int
    translated: str


class VertexTranslator(BaseTranslator):
    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        logger.info(
            f"Translating {len(texts)} subtitle shards using Vertex AI (Gemini Flash)..."
        )

        # Initialize the Vertex AI client using Application Default Credentials
        client = genai.Client(
            vertexai=True, project=self.project_id, location="us-central1"
        )

        # Package the texts into a JSON array of objects
        payload = [{"id": i, "text": t} for i, t in enumerate(texts)]
        payload_str = json.dumps(payload, ensure_ascii=False, indent=2)

        # Build dynamic prompt
        prompt = (
            f"You are a professional subtitle translator. "
            f"Translate the following JSON array of subtitle lines from {self.source_lang} to {self.target_lang}.\n"
            f"IMPORTANT RULES:\n"
            f"1. Maintain the exact same number of items and return a JSON array with 'id' and 'translated' fields.\n"
            f"2. The input lines are sequential parts of continuous speech.\n"
            f"3. Do not translate each line in strict isolation. Understand the complete thought across consecutive lines.\n"
            f"4. Because speakers naturally pause mid-sentence, a single sentence may be split across multiple lines. Translate the complete thought naturally, and distribute the English sentence across the corresponding lines so the grammar flows perfectly from one line to the next.\n"
        )
        if self.system_prompt:
            prompt += f"System Instructions: {self.system_prompt}\n"

        prompt += f"\nInput JSON:\n{payload_str}"

        # Use Gemini 2.5 Flash with structured output schema
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[TranslatedSubtitle],
                temperature=0.1,  # Low temperature to avoid hallucination inside JSON
            ),
        )

        try:
            # Parse the structured JSON response
            if not response.text:
                raise ValueError("LLM returned an empty response string.")

            response_json = json.loads(response.text)

            # Sort by id to guarantee ordering
            response_json.sort(key=lambda x: x["id"])

            translated_texts = [item["translated"] for item in response_json]

            if len(translated_texts) != len(texts):
                logger.warning(
                    "Warning: LLM returned a different number of translations than inputs. Subtitles may misalign."
                )

            return translated_texts

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw Response: {response.text}")
            raise e
