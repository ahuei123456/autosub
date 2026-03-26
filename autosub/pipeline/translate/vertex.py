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
    def _get_system_instruction(self, num_lines: int) -> str:
        instruction = (
            f"You are a professional subtitle translator and localizer.\n"
            f"Task: Translate the following JSON array of subtitle lines from {self.source_lang} to {self.target_lang}.\n\n"
            f"Output requirements:\n"
            f"1. Return valid JSON only.\n"
            f"2. You must return exactly {num_lines} items in the json, the same number of lines as the input.\n"
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
            f"8. If a Glossary is provided in the context, you MUST use the exact English translations specified for those terms.\n"
            f"9. Prefer ending subtitle lines on natural punctuation whenever possible. If a clause can end with a comma, period, question mark, or exclamation point, keep that punctuation at the end of the line instead of leaving a dangling conjunction or connective there.\n"
            f"10. Move trailing connectives such as 'but', 'and', 'so', 'because', 'though', or 'then' onto the following line.\n"
            f"11. Only use single quotes for contractions. Anywhere else, always use double quotation marks.\n"
            f"12. If a name is a Japanese name, preserve Japanese name order when translating.\n"
        )
        if self.system_prompt:
            instruction += (
                f"\nSpeaker and style context:\n{self.system_prompt.strip()}\n"
            )
        return instruction

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        logger.info(
            f"Translating {len(texts)} subtitle shards using Vertex AI (Gemini Flash)..."
        )

        # Initialize the Vertex AI client using Application Default Credentials
        client = genai.Client(vertexai=True, project=self.project_id, location="global")

        system_instruction = self._get_system_instruction(len(texts))
        payload = [{"id": i, "text": t} for i, t in enumerate(texts)]
        contents = json.dumps(payload, ensure_ascii=False, indent=2)

        # Use Gemini 3 Flash with structured output schema
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
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
