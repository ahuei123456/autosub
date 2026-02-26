from abc import ABC, abstractmethod


class BaseTranslator(ABC):
    def __init__(
        self,
        project_id: str,
        target_lang: str = "en",
        source_lang: str = "ja",
        system_prompt: str | None = None,
    ):
        self.project_id = project_id
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.system_prompt = system_prompt

    @abstractmethod
    def translate(self, texts: list[str]) -> list[str]:
        """
        Translates a list of strings and returns the translated list in the exact same order.
        """
        pass
