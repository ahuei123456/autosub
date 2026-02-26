import logging
from pathlib import Path

import pyass
from autosub.core.config import PROJECT_ID

logger = logging.getLogger(__name__)


def translate_subtitles(
    input_ass_path: Path,
    output_ass_path: Path,
    engine: str = "vertex",
    system_prompt: str | None = None,
    target_lang: str = "en",
    source_lang: str = "ja",
    bilingual: bool = True,
) -> None:
    """
    Reads an original .ass file, translates the dialogue events, and outputs a new .ass file.

    Args:
        input_ass_path: Path to the original (e.g., Japanese) .ass file.
        output_ass_path: Path to save the translated .ass file.
        engine: Translation engine ('vertex' or 'cloud-v3').
        system_prompt: Optional instructions for the LLM.
        target_lang: Language code to translate to.
        source_lang: Original language code.
        bilingual: If True, keep the original text above the translated text. If False, replace completely.
    """
    logger.info(f"Loading '{input_ass_path}' for translation...")

    with open(input_ass_path, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    # Extract only the dialogue text
    # We maintain a reference list to easily write the translations back to the correct events
    events_to_translate = []
    texts_to_translate = []

    for event in script.events:
        # standard Dialogue events or non-commented events
        if isinstance(event, pyass.Event) and event.text:
            # We don't want to translate raw .ass tags.
            # In a robust implementation, we'd strip {\\tags} before translating.
            # Pyass has an event.text property which returns the raw text. Let's grab the raw string representation of parts.
            raw_text = event.text

            if raw_text.strip():
                events_to_translate.append(event)
                texts_to_translate.append(raw_text)

    if not texts_to_translate:
        logger.warning("No subtitle text found to translate. Exiting.")
        return

    if not PROJECT_ID:
        raise ValueError("AUTOSUB_PROJECT_ID is not set in the environment.")

    if engine == "vertex":
        from autosub.pipeline.translate.vertex import VertexTranslator

        translator = VertexTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
        )
    elif engine == "cloud-v3":
        from autosub.pipeline.translate.api import CloudTranslationTranslator

        translator = CloudTranslationTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unknown translation engine: {engine}")

    translated_texts = translator.translate(texts_to_translate)

    if len(translated_texts) != len(events_to_translate):
        raise ValueError(
            "Translation API returned a different number of results than expected."
        )

    logger.info("Applying translations to subtitle events...")

    # Update the events with the new text
    for i, event in enumerate(events_to_translate):
        original_text = texts_to_translate[i]
        translated_text = translated_texts[i]

        if bilingual:
            # Create a stacked visual format: Original (small/translucent) on top, Translated (large) on bottom
            # Formatting uses standard ASS override tags
            new_text_str = f"{{\\\\fs24\\\\a6}}{original_text}{{\\\\N}}{{\\\\fs48\\\\a2}}{translated_text}"

            # Since Pyass parses parts, we assign a new EventText part containing our constructed string.
            # Technically, Pyass will            # overide tags as standard text if we don't parse them,
            # but Aegisub/video players handle raw string output perfectly fine.
            event.text = new_text_str
        else:
            event.text = translated_text

    logger.info(f"Writing translated .ass file to {output_ass_path}...")
    with open(output_ass_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)

    logger.info("Translation complete!")
