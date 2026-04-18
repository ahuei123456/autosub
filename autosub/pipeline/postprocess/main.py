from __future__ import annotations

import logging
import re
from pathlib import Path

import pyass

logger = logging.getLogger(__name__)

BILINGUAL_TRANSLATION_TAG = r"{\fs48\a2}"
QUOTE_CHARS = {'"', "“", "”"}
LINE_BREAK_RE = re.compile(r"(\\N|\\n|\r\n|\n|\r)")


def postprocess_subtitles(
    input_ass_path: Path,
    output_ass_path: Path | None = None,
    extensions_config: dict | None = None,
    bilingual: bool = True,
) -> None:
    if output_ass_path is None:
        output_ass_path = input_ass_path

    if not extensions_config:
        return

    with open(input_ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    modified = False

    radio_discourse_config = extensions_config.get("radio_discourse", {})
    if radio_discourse_config.get("enabled"):
        modified = _apply_radio_discourse_postprocess(script, bilingual) or modified

    if not modified:
        return

    logger.info(f"Writing postprocessed subtitles to {output_ass_path}...")
    with open(output_ass_path, "w", encoding="utf-8") as handle:
        pyass.dump(script, handle)


def _apply_radio_discourse_postprocess(script: pyass.Script, bilingual: bool) -> bool:
    modified = False
    for event in script.events:
        if not isinstance(event, pyass.Event) or not event.text:
            continue
        if event.name != "listener_mail":
            continue

        quoted = _quote_listener_mail_text(event.text, bilingual=bilingual)
        if quoted != event.text:
            event.text = quoted
            modified = True
    return modified


def _quote_listener_mail_text(text: str, bilingual: bool) -> str:
    if bilingual and BILINGUAL_TRANSLATION_TAG in text:
        prefix, translated = text.rsplit(BILINGUAL_TRANSLATION_TAG, 1)
        return f"{prefix}{BILINGUAL_TRANSLATION_TAG}{_ensure_quoted(translated)}"

    return _ensure_quoted(text)


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
