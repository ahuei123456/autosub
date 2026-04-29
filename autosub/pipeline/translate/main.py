import hashlib
import json
import logging
import traceback
from pathlib import Path

from autosub.core.config import PROJECT_ID
from autosub.core.llm import ReasoningEffort
from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.pipeline.format.generator import render_ass_document
from autosub.pipeline.translate.chunker import make_chunks

logger = logging.getLogger(__name__)


def _compute_cue_fingerprint(
    cues: list[SubtitleCue], chunk_size: int, corner_boundaries: list[int] | None
) -> str:
    """Hash translation inputs and cue metadata to detect stale checkpoints."""
    h = hashlib.sha256()
    h.update(str(chunk_size).encode())
    h.update(b"\x00")
    for b in corner_boundaries or []:
        h.update(str(b).encode())
        h.update(b"\x00")
    h.update(b"\x01")
    for cue in cues:
        source_text = cue.normalized_source_text or cue.source_text
        payload = {
            "text": source_text,
            "start_time": cue.start_time,
            "end_time": cue.end_time,
            "speaker": cue.speaker,
            "role": cue.role,
            "corner": cue.corner,
        }
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()


def translate_subtitles(
    input_json_path: Path,
    output_json_path: Path,
    output_ass_path: Path | None = None,
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
    Reads a formatted subtitle JSON document, translates cue source text, and
    writes a translated JSON document plus a rendered ASS byproduct.

    Corner boundaries are read directly from structured cue metadata.
    """
    if output_ass_path is None:
        output_ass_path = output_json_path.with_suffix(".ass")

    logger.info(f"Loading '{input_json_path}' for translation...")
    document = SubtitleDocument.model_validate_json(
        input_json_path.read_text(encoding="utf-8")
    )

    cues_to_translate = []
    texts_to_translate = []
    for cue in document.cues:
        source_text = cue.normalized_source_text or cue.source_text
        if source_text.strip():
            cues_to_translate.append(cue)
            texts_to_translate.append(source_text)

    if not texts_to_translate:
        logger.warning("No subtitle text found to translate. Exiting.")
        translated_document = document.model_copy(deep=True)
        translated_document.stage = "translated"
        output_json_path.write_text(
            translated_document.model_dump_json(indent=2), encoding="utf-8"
        )
        render_ass_document(
            translated_document,
            output_ass_path,
            mode="bilingual" if bilingual else "translated",
        )
        return

    llm_trace_path: Path | None = None

    if engine == "vertex":
        from autosub.pipeline.translate.translator import VertexTranslator

        if provider in {"google-vertex", "anthropic-vertex"} and not PROJECT_ID:
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

    checkpoint_path = output_json_path.with_suffix(".checkpoint.json")
    error_path = output_json_path.with_suffix(".error.txt")

    if error_path.exists():
        error_path.unlink()
        logger.info("Removed previous translation error file.")

    corner_boundaries = _extract_corner_boundaries_from_cues(document)
    if corner_boundaries:
        logger.info(
            f"Found {len(corner_boundaries)} corner boundaries at dialogue indices {corner_boundaries}"
        )

    splits: set[int] = set()
    try:
        if chunk_size > 0:
            translated_texts, splits = _translate_chunked(
                translator,
                cues_to_translate,
                chunk_size,
                checkpoint_path,
                corner_boundaries=corner_boundaries or None,
                retry_chunks=retry_chunks,
                log_dir=log_dir,
            )
        else:
            translated_texts = translator.translate_cues(cues_to_translate)
    except Exception as exc:
        _write_error_report(error_path, exc)
        logger.error(f"Wrote translation error details to {error_path}.")
        raise

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file.")

    if len(translated_texts) != len(cues_to_translate):
        raise ValueError(
            f"Translation API expected {len(cues_to_translate)} translations, but got {len(translated_texts)}"
        )

    logger.info("Applying translations to subtitle document...")
    translated_document = document.model_copy(deep=True)
    translated_document.stage = "translated"
    cue_by_id = {cue.id: cue for cue in translated_document.cues}
    for source_cue, translated_text in zip(
        cues_to_translate, translated_texts, strict=True
    ):
        cue_by_id[source_cue.id].translated_text = translated_text

    logger.info(f"Writing translated JSON to {output_json_path}...")
    output_json_path.write_text(
        translated_document.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info(f"Writing translated .ass file to {output_ass_path}...")
    render_ass_document(
        translated_document,
        output_ass_path,
        mode="bilingual" if bilingual else "translated",
        chunk_boundaries=splits if debug else None,
    )

    if llm_trace_path is not None and llm_trace_path.exists():
        logger.info(f"Wrote LLM trace to {llm_trace_path}.")

    logger.info("Translation complete!")


def _extract_corner_boundaries_from_cues(document: SubtitleDocument) -> list[int]:
    boundaries: list[int] = []
    dialogue_idx = 0
    for cue in document.cues:
        source_text = cue.normalized_source_text or cue.source_text
        if not source_text.strip():
            if cue.corner:
                logger.warning(
                    "Ignoring corner boundary on empty cue %s.",
                    cue.id,
                )
            continue
        if cue.corner:
            boundaries.append(dialogue_idx)
        dialogue_idx += 1
    return boundaries


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
    cues: list[SubtitleCue],
    chunk_size: int,
    checkpoint_path: Path,
    corner_boundaries: list[int] | None = None,
    retry_chunks: list[int] | None = None,
    log_dir: Path | None = None,
) -> tuple[list[str], set[int]]:
    """Split texts into chunks, translate each once, and merge results."""
    texts = [cue.normalized_source_text or cue.source_text for cue in cues]
    chunks, splits = make_chunks(texts, chunk_size, corner_boundaries=corner_boundaries)
    fingerprint = _compute_cue_fingerprint(cues, chunk_size, corner_boundaries)

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
        cue_chunk = cues[line_offset : line_offset + len(chunk)]
        if len(cue_chunk) != len(chunk):
            raise ValueError("Cue chunking lost alignment with text chunks.")

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
        results = translator.translate_cues(cue_chunk)
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
