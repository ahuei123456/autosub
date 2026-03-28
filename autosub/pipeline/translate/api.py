import logging
from google.cloud import translate

from autosub.pipeline.translate.base import BaseTranslator

logger = logging.getLogger(__name__)


class CloudTranslationTranslator(BaseTranslator):
    def __init__(
        self,
        *,
        project_id: str,
        target_lang: str = "en",
        source_lang: str = "ja",
        system_prompt: str | None = None,
    ):
        super().__init__(
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
        )
        self.project_id = project_id

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        client = translate.TranslationServiceClient()
        parent = f"projects/{self.project_id}/locations/global"

        logger.info(
            f"Translating {len(texts)} subtitle shards using standard Cloud Translation API..."
        )

        if self.system_prompt:
            logger.warning(
                "CloudTranslationTranslator does not support custom system prompts. Ignoring '--prompt'."
            )

        response = client.translate_text(
            request={
                "parent": parent,
                "contents": texts,
                "mime_type": "text/plain",
                "source_language_code": self.source_lang,
                "target_language_code": self.target_lang,
            }
        )

        return [translation.translated_text for translation in response.translations]
