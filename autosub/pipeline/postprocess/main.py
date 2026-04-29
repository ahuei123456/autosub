from __future__ import annotations

import logging
import re
from pathlib import Path

from autosub.core.schemas import SubtitleDocument
from autosub.pipeline.format.generator import render_ass_document

logger = logging.getLogger(__name__)

QUOTE_CHARS = {'"', "“", "”"}
LINE_BREAK_RE = re.compile(r"(\\N|\\n|\r\n|\n|\r)")


def postprocess_subtitles(
    input_json_path: Path,
    output_json_path: Path | None = None,
    output_ass_path: Path | None = None,
    extensions_config: dict | None = None,
    bilingual: bool = True,
) -> None:
    if output_json_path is None:
        output_json_path = input_json_path.with_name("postprocessed.json")
    if output_ass_path is None:
        output_ass_path = output_json_path.with_suffix(".ass")

    document = SubtitleDocument.model_validate_json(
        input_json_path.read_text(encoding="utf-8")
    )
    processed = document.model_copy(deep=True)
    processed.stage = "postprocessed"
    for cue in processed.cues:
        cue.final_text = cue.final_text or cue.translated_text

    extensions_config = extensions_config or {}
    radio_discourse_config = extensions_config.get("radio_discourse", {})
    if radio_discourse_config.get("enabled"):
        if _apply_radio_discourse_postprocess(processed):
            logger.info("Postprocessing modified subtitle document.")

    logger.info(f"Writing postprocessed JSON to {output_json_path}...")
    output_json_path.write_text(processed.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"Writing postprocessed subtitles to {output_ass_path}...")
    render_ass_document(
        processed, output_ass_path, mode="bilingual" if bilingual else "final"
    )


def _apply_radio_discourse_postprocess(document: SubtitleDocument) -> bool:
    modified = False
    for cue in document.cues:
        if cue.role != "listener_mail":
            continue
        if not cue.final_text:
            continue

        quoted = _ensure_quoted(cue.final_text)
        if quoted != cue.final_text:
            cue.final_text = quoted
            modified = True
    return modified


def _ensure_quoted(text: str) -> str:
    if _is_wrapped_in_quotes(text):
        return _normalize_quote_edges(text)
    return _normalize_quote_edges(f'"{text}"')


def _is_wrapped_in_quotes(text: str) -> bool:
    stripped = text.strip()
    return (
        len(stripped) >= 2
        and stripped[0] in QUOTE_CHARS
        and stripped[-1] in QUOTE_CHARS
    )


def _collapse_outer_duplicate_quotes(text: str) -> str:
    leading_whitespace_len = len(text) - len(text.lstrip())
    trailing_whitespace_len = len(text) - len(text.rstrip())
    trailing_start = (
        len(text) - trailing_whitespace_len if trailing_whitespace_len else len(text)
    )

    prefix = text[:leading_whitespace_len]
    core = text[leading_whitespace_len:trailing_start]
    suffix = text[trailing_start:]

    while (
        len(core) >= 4
        and core[0] in QUOTE_CHARS
        and core[1] in QUOTE_CHARS
        and core[-1] in QUOTE_CHARS
        and core[-2] in QUOTE_CHARS
    ):
        core = core[1:-1]

    return f"{prefix}{core}{suffix}"


def _normalize_quote_edges(text: str) -> str:
    collapsed = _collapse_outer_duplicate_quotes(text)
    return _collapse_duplicate_visual_line_quotes(collapsed)


def _collapse_duplicate_visual_line_quotes(text: str) -> str:
    parts = LINE_BREAK_RE.split(text)
    if not parts:
        return text

    # Only collapse the display edges of a multi-line cue; duplicated quotes next
    # to interior line breaks are preserved because they belong to inner content.
    parts[0] = _collapse_leading_quote_run(parts[0])
    parts[-1] = _collapse_trailing_quote_run(parts[-1])
    return "".join(parts)


def _collapse_leading_quote_run(text: str) -> str:
    leading_whitespace_len = len(text) - len(text.lstrip())
    prefix = text[:leading_whitespace_len]
    core = text[leading_whitespace_len:]

    while len(core) >= 2 and core[0] in QUOTE_CHARS and core[1] in QUOTE_CHARS:
        core = core[1:]

    return f"{prefix}{core}"


def _collapse_trailing_quote_run(text: str) -> str:
    trailing_whitespace_len = len(text) - len(text.rstrip())
    suffix = text[-trailing_whitespace_len:] if trailing_whitespace_len else ""
    core = (
        text[: len(text) - trailing_whitespace_len] if trailing_whitespace_len else text
    )

    while len(core) >= 2 and core[-1] in QUOTE_CHARS and core[-2] in QUOTE_CHARS:
        core = core[:-1]

    return f"{core}{suffix}"
