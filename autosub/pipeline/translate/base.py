from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autosub.core.schemas import SubtitleCue


class BaseTranslator(ABC):
    def __init__(
        self,
        *,
        target_lang: str = "en",
        source_lang: str = "ja",
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.system_prompt = system_prompt

    @abstractmethod
    def translate(self, texts: list[str]) -> list[str]:
        """
        Translates a list of strings and returns the translated list in the exact same order.
        """
        pass

    def translate_cues(self, cues: list["SubtitleCue"]) -> list[str]:
        texts = [cue.normalized_source_text or cue.source_text for cue in cues]
        return self.translate(texts)
