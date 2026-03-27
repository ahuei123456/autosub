import json
import logging
import time
from pathlib import Path

import pyass
from autosub.core.config import PROJECT_ID

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds


def translate_subtitles(
    input_ass_path: Path,
    output_ass_path: Path,
    engine: str = "vertex",
    system_prompt: str | None = None,
    target_lang: str = "en",
    source_lang: str = "ja",
    bilingual: bool = True,
    model: str = "gemini-3-flash-preview",
    location: str = "global",
    chunk_size: int = 0,
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
        model: Vertex model name for LLM translation.
        location: Vertex region for LLM translation.
        chunk_size: If > 0, split into chunks of this size with retry logic.
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
        from autosub.pipeline.translate.translator import VertexTranslator

        translator = VertexTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
            model=model,
            location=location,
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

    checkpoint_path = output_ass_path.with_suffix(".checkpoint.json")

    if chunk_size > 0:
        translated_texts = _translate_chunked(
            translator, texts_to_translate, chunk_size, checkpoint_path
        )
    else:
        translated_texts = _translate_with_retry(translator, texts_to_translate)

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file.")

    if len(translated_texts) != len(events_to_translate):
        raise ValueError(
            f"Translation API expected {len(events_to_translate)} translations, but got {len(translated_texts)}"
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


def _translate_with_retry(translator, texts: list[str], label: str = "") -> list[str]:
    """Translate texts with exponential backoff retry."""
    prefix = f"  {label}: " if label else ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return translator.translate(texts)
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"{prefix}Attempt {attempt} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{prefix}Failed after {MAX_RETRIES} attempts: {e}"
                )
                raise
    return []  # unreachable


def _load_checkpoint(checkpoint_path: Path) -> dict[int, list[str]]:
    """Load and validate completed chunk results from checkpoint file.

    Expected structure: {"0": ["str", ...], "1": ["str", ...], ...}
    Keys must be non-negative integer strings, values must be non-empty lists of strings.
    Invalid entries are skipped with a warning.
    """
    if not checkpoint_path.exists():
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint, starting fresh: {e}")
        return {}

    if not isinstance(data, dict):
        logger.warning(f"Checkpoint is not a JSON object, starting fresh.")
        return {}

    validated: dict[int, list[str]] = {}
    for k, v in data.items():
        try:
            chunk_idx = int(k)
        except (ValueError, TypeError):
            logger.warning(f"Skipping checkpoint entry with non-integer key: {k!r}")
            continue

        if chunk_idx < 0:
            logger.warning(f"Skipping checkpoint entry with negative key: {chunk_idx}")
            continue

        if not isinstance(v, list) or not v:
            logger.warning(f"Skipping checkpoint entry {chunk_idx}: value must be a non-empty list.")
            continue

        if not all(isinstance(s, str) for s in v):
            logger.warning(f"Skipping checkpoint entry {chunk_idx}: list contains non-string elements.")
            continue

        validated[chunk_idx] = v

    return validated


def _save_checkpoint(
    checkpoint_path: Path, completed: dict[int, list[str]]
) -> None:
    """Save completed chunk results to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False, indent=2)


def _translate_chunked(
    translator, texts: list[str], chunk_size: int, checkpoint_path: Path
) -> list[str]:
    """Split texts into chunks, translate each with retry logic, and merge results."""
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    completed = _load_checkpoint(checkpoint_path)

    if completed:
        logger.info(
            f"Resuming from checkpoint: {len(completed)}/{len(chunks)} chunks already completed."
        )

    logger.info(
        f"Translating {len(texts)} subtitle lines "
        f"in {len(chunks)} chunks of up to {chunk_size}..."
    )

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx in completed:
            logger.info(
                f"  Chunk {chunk_idx + 1}/{len(chunks)} — skipped (checkpoint)"
            )
            continue

        logger.info(
            f"  Chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} lines)..."
        )
        results = _translate_with_retry(
            translator, chunk, label=f"Chunk {chunk_idx + 1}"
        )
        completed[chunk_idx] = results
        _save_checkpoint(checkpoint_path, completed)

    # Reassemble in order
    all_translated: list[str] = []
    for chunk_idx in range(len(chunks)):
        all_translated.extend(completed[chunk_idx])

    return all_translated
