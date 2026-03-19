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
            f"You are a professional subtitle translator and localizer.\n"
            f"Task: Translate the following JSON array of subtitle lines from {self.source_lang} to {self.target_lang}.\n\n"
            f"Output requirements:\n"
            f"1. Return valid JSON only.\n"
            f"2. Return the exact same number of items as the input.\n"
            f"3. Each item must contain exactly two fields: 'id' and 'translated'.\n"
            f"4. Preserve item order and keep each translation matched to its original id.\n\n"
            f"Translation requirements:\n"
            f"1. The input lines are sequential pieces of continuous spoken dialogue.\n"
            f"2. Do not translate each line in isolation. Understand the thought across neighboring lines before deciding wording.\n"
            f"3. If one sentence is split across multiple subtitle lines, translate the full thought naturally and distribute the English across those lines so the flow still reads naturally line by line.\n"
            f"4. Prioritize natural subtitle English over literal wording, but do not invent information.\n"
            f"5. Keep the tone, emotional intent, and speaker persona intact.\n"
            f"6. Keep translations concise enough to work as readable subtitles.\n"
            f"7. If a line contains a catchphrase, segment title, fandom reference, or recurring term, translate it consistently with the provided context.\n"
        )
        if self.system_prompt:
            prompt += f"\nSpeaker and style context:\n{self.system_prompt.strip()}\n"

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
