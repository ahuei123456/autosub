import hashlib
import json
import logging
import time
import traceback
from pathlib import Path

import pyass
from autosub.core.config import PROJECT_ID
from autosub.core.llm import ReasoningEffort
from autosub.pipeline.translate.chunker import make_chunks

logger = logging.getLogger(__name__)


def _compute_fingerprint(
    texts: list[str], chunk_size: int, corner_boundaries: list[int] | None
) -> str:
    """Hash input texts and chunking config to detect stale checkpoints."""
    h = hashlib.sha256()
    h.update(str(chunk_size).encode())
    h.update(b"\x00")
    for b in corner_boundaries or []:
        h.update(str(b).encode())
        h.update(b"\x00")
    h.update(b"\x01")
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _extract_corner_boundaries(
    all_events: list[pyass.Event],
    events_to_translate: list[pyass.Event],
) -> list[int]:
    """Extract corner boundary indices from Comment events in the ASS script.

    Corner Comment events (effect="corner") are placed by the format-time
    corners extension. This function maps each corner Comment to the index
    of the next dialogue event in the translate list, giving the chunker
    pre-computed boundary positions.
    """
    translate_set = set(id(e) for e in events_to_translate)
    boundaries: list[int] = []
    pending_corner = False
    dialogue_idx = 0

    for event in all_events:
        if (
            isinstance(event, pyass.Event)
            and event.format == pyass.EventFormat.COMMENT
            and event.effect == "corner"
        ):
            # A corner annotation attaches to the *next* translatable dialogue
            # event, not necessarily the immediately following event. This
            # handles intervening non-translatable Comments (e.g. chunk markers)
            # that may appear between the corner Comment and its target dialogue.
            pending_corner = True
            continue

        if id(event) in translate_set:
            if pending_corner:
                boundaries.append(dialogue_idx)
                pending_corner = False
            dialogue_idx += 1

    return boundaries


def translate_subtitles(
    input_ass_path: Path,
    output_ass_path: Path,
    engine: str = "vertex",
    system_prompt: str | None = None,
    target_lang: str = "en",
    source_lang: str = "ja",
    bilingual: bool = True,
    model: str | None = None,
    location: str = "global",
    provider: str = "google-vertex",
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
    reasoning_budget_tokens: int | None = None,
    reasoning_dynamic: bool | None = None,
    chunk_size: int = 0,
    debug: bool = False,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
) -> None:
    """
    Reads an original .ass file, translates the dialogue events, and outputs a new .ass file.

    Corner boundaries are automatically extracted from Comment events with
    effect="corner" in the input ASS (placed by the corners format extension).
    """
    logger.info(f"Loading '{input_ass_path}' for translation...")

    with open(input_ass_path, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    # Extract only the dialogue text
    # We maintain a reference list to easily write the translations back to the correct events
    events_to_translate = []
    texts_to_translate = []

    for event in script.events:
        # Skip Comment events so they aren't sent to the LLM
        if isinstance(event, pyass.Event) and event.format == pyass.EventFormat.COMMENT:
            continue
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

    llm_trace_path: Path | None = None

    if engine == "vertex":
        from autosub.pipeline.translate.translator import VertexTranslator

        if provider == "google-vertex" and not PROJECT_ID:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in the environment.")

        llm_trace_path = output_ass_path.with_suffix(".llm_trace.jsonl")
        if llm_trace_path.exists():
            llm_trace_path.unlink()
            logger.info("Removed previous LLM trace file.")

        translator = VertexTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
            model=model,
            location=location,
            provider=provider,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_dynamic=reasoning_dynamic,
            trace_path=llm_trace_path,
        )
    elif engine == "cloud-v3":
        from autosub.pipeline.translate.api import CloudTranslationTranslator

        if not PROJECT_ID:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in the environment.")

        translator = CloudTranslationTranslator(
            project_id=PROJECT_ID,
            target_lang=target_lang,
            source_lang=source_lang,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unknown translation engine: {engine}")

    checkpoint_path = output_ass_path.with_suffix(".checkpoint.json")
    error_path = output_ass_path.with_suffix(".error.txt")

    if error_path.exists():
        error_path.unlink()
        logger.info("Removed previous translation error file.")

    # Extract corner boundaries from Comment events placed by format-time extension
    corner_boundaries = _extract_corner_boundaries(script.events, events_to_translate)
    if corner_boundaries:
        logger.info(
            f"Found {len(corner_boundaries)} corner boundaries at dialogue indices {corner_boundaries}"
        )

    splits: set[int] = set()
    try:
        if chunk_size > 0:
            translated_texts, splits = _translate_chunked(
                translator, texts_to_translate, chunk_size, checkpoint_path,
                corner_boundaries=corner_boundaries or None,
                retry_chunks=retry_chunks,
                log_dir=log_dir,
            )
        else:
            translated_texts = translator.translate(texts_to_translate)
    except Exception as exc:
        _write_error_report(error_path, exc)
        logger.error(f"Wrote translation error details to {error_path}.")
        raise

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file.")

    if len(translated_texts) != len(events_to_translate):
        raise ValueError(
            f"Translation API expected {len(events_to_translate)} translations, but got {len(translated_texts)}"
        )

    logger.info("Applying translations to subtitle events...")

    new_events: list[pyass.Event] = []
    translated_event_set = set(id(e) for e in events_to_translate)

    # Walk all events in order, preserving non-translated events (e.g. corner Comments)
    # in place while applying translations.
    event_idx = 0
    for event in script.events:
        if id(event) not in translated_event_set:
            # Non-dialogue event (Comment, etc.) — keep as-is
            new_events.append(event)
            continue

        # Match this event to its translation by index
        original_text = texts_to_translate[event_idx]
        translated_text = translated_texts[event_idx]

        # Insert debug comment at artificial chunk boundaries
        if debug and event_idx in splits:
            debug_comment = pyass.Event(
                format=pyass.EventFormat.COMMENT,
                start=event.start,
                end=event.end,
                style=event.style,
                effect="",
                text="[autosub] Chunk boundary — review translation around this line",
            )
            new_events.append(debug_comment)

        event_idx += 1

        # Update the event with the new text
        if bilingual:
            event.text = f"{{\\\\fs24\\\\a6}}{original_text}{{\\\\N}}{{\\\\fs48\\\\a2}}{translated_text}"
        else:
            event.text = translated_text

        new_events.append(event)

    script.events = new_events

    logger.info(f"Writing translated .ass file to {output_ass_path}...")
    with open(output_ass_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)

    if llm_trace_path is not None and llm_trace_path.exists():
        logger.info(f"Wrote LLM trace to {llm_trace_path}.")

    logger.info("Translation complete!")


def _write_error_report(error_path: Path, exc: Exception) -> None:
    error_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )


def _load_checkpoint(checkpoint_path: Path, fingerprint: str) -> dict[int, list[str]]:
    """Load and validate completed chunk results from checkpoint file.

    Returns dict[int, list[str]] mapping chunk index to translated strings.
    JSON serializes int keys as strings, so they are converted back on load.
    Invalid entries are skipped with a warning.
    Discards the checkpoint if the fingerprint doesn't match (input changed).
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
        logger.warning("Checkpoint is not a JSON object, starting fresh.")
        return {}

    # Validate fingerprint
    if "_fingerprint" not in data:
        logger.warning("Legacy checkpoint without fingerprint, discarding.")
        return {}
    if data["_fingerprint"] != fingerprint:
        logger.warning(
            "Checkpoint fingerprint mismatch (input or chunking config changed), "
            "discarding stale checkpoint."
        )
        return {}

    chunks_data = data.get("chunks", {})
    if not isinstance(chunks_data, dict):
        logger.warning("Checkpoint 'chunks' is not a dict, starting fresh.")
        return {}

    validated: dict[int, list[str]] = {}
    for k, v in chunks_data.items():
        try:
            chunk_idx = int(k)
        except (ValueError, TypeError):
            logger.warning(f"Skipping checkpoint entry with non-integer key: {k!r}")
            continue

        if chunk_idx < 0:
            logger.warning(f"Skipping checkpoint entry with negative key: {chunk_idx}")
            continue

        if not isinstance(v, list) or not v:
            logger.warning(
                f"Skipping checkpoint entry {chunk_idx}: value must be a non-empty list."
            )
            continue

        if not all(isinstance(s, str) for s in v):
            logger.warning(
                f"Skipping checkpoint entry {chunk_idx}: list contains non-string elements."
            )
            continue

        validated[chunk_idx] = v

    return validated


def _save_checkpoint(
    checkpoint_path: Path, completed: dict[int, list[str]], fingerprint: str
) -> None:
    """Save completed chunk results to checkpoint file."""
    payload = {"_fingerprint": fingerprint, "chunks": completed}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _translate_chunked(
    translator,
    texts: list[str],
    chunk_size: int,
    checkpoint_path: Path,
    corner_boundaries: list[int] | None = None,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
) -> tuple[list[str], set[int]]:
    """Split texts into chunks, translate each once, and merge results."""
    chunks, splits = make_chunks(texts, chunk_size, corner_boundaries=corner_boundaries)
    fingerprint = _compute_fingerprint(texts, chunk_size, corner_boundaries)

    # Set up structured log directory
    chunks_dir = None
    token_summary_path = None
    system_prompt_path = None
    if log_dir:
        chunks_dir = log_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        token_summary_path = log_dir / "token_summary.tsv"
        system_prompt_path = log_dir / "system_prompt.txt"
        # Write header if new file
        if not token_summary_path.exists():
            token_summary_path.write_text(
                "chunk\tlines\tprompt\tcandidates\tthoughts\ttotal\n",
                encoding="utf-8",
            )
    completed = _load_checkpoint(checkpoint_path, fingerprint)

    # Remove specified chunks from checkpoint to force re-translation
    if retry_chunks and completed:
        for idx in retry_chunks:
            chunk_num = idx - 1  # user-facing is 1-based
            if chunk_num in completed:
                del completed[chunk_num]
                logger.info(f"Cleared checkpoint for chunk {idx} — will re-translate.")
            else:
                logger.warning(f"Chunk {idx} not in checkpoint — nothing to retry.")
        _save_checkpoint(checkpoint_path, completed, fingerprint)

    if completed:
        logger.info(
            f"Resuming from checkpoint: {len(completed)}/{len(chunks)} chunks already completed."
        )

    logger.info(
        f"Translating {len(texts)} subtitle lines "
        f"in {len(chunks)} chunks of up to {chunk_size}..."
    )

    line_offset = 0
    for chunk_idx, chunk in enumerate(chunks):
        line_start = line_offset + 1
        line_end = line_offset + len(chunk)

        if chunk_idx in completed:
            logger.info(f"  Chunk {chunk_idx + 1}/{len(chunks)} — skipped (checkpoint)")
            line_offset += len(chunk)
            continue

        first = chunk[0][:40] + "..." if len(chunk[0]) > 40 else chunk[0]
        last = chunk[-1][:40] + "..." if len(chunk[-1]) > 40 else chunk[-1]
        logger.info(
            f"  Chunk {chunk_idx + 1}/{len(chunks)} "
            f"(lines {line_start}-{line_end}, {len(chunk)} lines)"
        )
        logger.info(f"    first: {first}")
        logger.info(f"    last:  {last}")
        results = translator.translate(chunk)
        completed[chunk_idx] = results
        _save_checkpoint(checkpoint_path, completed, fingerprint)

        # Write structured log files per chunk
        if chunks_dir and hasattr(translator, "last_diagnostics"):
            chunk_num = f"{chunk_idx + 1:02d}"

            # Write system prompt once
            if system_prompt_path and (
                chunk_idx == 0 or not system_prompt_path.exists()
            ):
                if hasattr(translator, "last_system_instruction"):
                    system_prompt_path.write_text(
                        translator.last_system_instruction, encoding="utf-8"
                    )

            if hasattr(translator, "last_input"):
                (chunks_dir / f"chunk_{chunk_num}_input.json").write_text(
                    translator.last_input, encoding="utf-8"
                )
            if hasattr(translator, "last_output"):
                (chunks_dir / f"chunk_{chunk_num}_output.json").write_text(
                    translator.last_output, encoding="utf-8"
                )

            diag = translator.last_diagnostics
            if diag.thinking_text:
                (chunks_dir / f"chunk_{chunk_num}_thinking.txt").write_text(
                    diag.thinking_text, encoding="utf-8"
                )

            if token_summary_path:
                with open(token_summary_path, "a", encoding="utf-8") as tsv:
                    tsv.write(
                        f"{chunk_idx + 1}\t{len(chunk)}\t"
                        f"{diag.prompt_token_count}\t{diag.candidates_token_count}\t"
                        f"{diag.thoughts_token_count}\t{diag.total_token_count}\n"
                    )

        line_offset += len(chunk)

    # Reassemble in order
    all_translated: list[str] = []
    for chunk_idx in range(len(chunks)):
        all_translated.extend(completed[chunk_idx])

    return all_translated, splits
