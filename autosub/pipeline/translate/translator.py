import logging
import json
from pathlib import Path

from pydantic import BaseModel

from autosub.core.errors import VertexResponseShapeError
from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.pipeline.translate.base import BaseTranslator

logger = logging.getLogger(__name__)


class TranslatedSubtitle(BaseModel):
    id: int
    translated: str


class VertexTranslator(BaseTranslator, BaseStructuredLLM):
    DEFAULT_MODELS = {
        "google-vertex": "gemini-3-flash-preview",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
        "openrouter": "openai/gpt-5-mini",
    }

    def __init__(
        self,
        *,
        project_id: str | None,
        target_lang: str = "en",
        source_lang: str = "ja",
        system_prompt: str | None = None,
        model: str | None = None,
        location: str = "global",
        temperature: float = 0.1,
        provider: str = "google-vertex",
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
        reasoning_budget_tokens: int | None = None,
        reasoning_dynamic: bool | None = None,
        provider_options: dict[str, object] | None = None,
        trace_path: Path | str | None = None,
    ):
        resolved_model = model or self.DEFAULT_MODELS.get(
            provider, "gemini-3-flash-preview"
        )
        super().__init__(
            project_id=project_id,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
            model=resolved_model,
            location=location,
            temperature=temperature,
            provider=provider,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_dynamic=reasoning_dynamic,
            provider_options=provider_options,
            trace_path=trace_path,
        )

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
            f"13. Do not use the em-dash character (—) in translations. When separating clauses, prefer to end the current sentence or use a comma to separate causes while making it sound natural in English. Use ellipsis only if the speaker is trailing off before starting the next line.\n"
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
            "Translating %s subtitle shards using provider '%s' model '%s' in '%s'...",
            len(texts),
            self.provider,
            self.model,
            self.location,
        )

        system_instruction = self._get_system_instruction(len(texts))
        payload = [{"id": i, "text": t} for i, t in enumerate(texts)]
        contents = json.dumps(payload, ensure_ascii=False, indent=2)

        translations, diagnostics = self._run_structured_output(
            user_prompt=contents,
            system_prompt=system_instruction,
            output_type=list[TranslatedSubtitle],
            operation_name="LLM translator",
            output_name="subtitle_translations",
        )

        # Store for structured logging by the caller
        self.last_system_instruction = system_instruction
        self.last_input = contents
        self.last_output = json.dumps(
            [t.model_dump() for t in translations], ensure_ascii=False, indent=2
        )
        self.last_diagnostics = diagnostics

        try:
            ordered_translations = sorted(translations, key=lambda item: item.id)
            translated_texts = [item.translated for item in ordered_translations]
            returned_ids = [item.id for item in ordered_translations]

            if returned_ids != list(range(len(texts))):
                raise ValueError(f"returned ids were {returned_ids!r}")

            if len(translated_texts) != len(texts):
                logger.warning(
                    "Warning: LLM returned a different number of translations than inputs. Subtitles may misalign."
                )

            return translated_texts

        except Exception as exc:
            raise VertexResponseShapeError(
                f"LLM translator returned JSON with an unexpected structure: {exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc
