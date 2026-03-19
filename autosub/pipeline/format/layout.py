from __future__ import annotations

import unicodedata

from autosub.core.schemas import SubtitleLine

PREFERRED_BREAK_CHARS = {
    " ",
    "　",
    "、",
    "。",
    "！",
    "？",
    "・",
    ",",
    ".",
    "!",
    "?",
}


def wrap_subtitle_lines(
    lines: list[SubtitleLine],
    max_line_width: int = 22,
    max_lines_per_subtitle: int = 2,
) -> list[SubtitleLine]:
    """
    Inserts ASS line breaks so a subtitle renders in at most `max_lines_per_subtitle`
    lines. The heuristic is tuned for Japanese text but remains safe for mixed text.
    """
    if max_lines_per_subtitle < 1:
        return lines

    wrapped_lines: list[SubtitleLine] = []
    for line in lines:
        wrapped_lines.append(
            SubtitleLine(
                text=_wrap_text(
                    line.text,
                    max_line_width=max_line_width,
                    max_lines=max_lines_per_subtitle,
                ),
                start_time=line.start_time,
                end_time=line.end_time,
                speaker=line.speaker,
            )
        )

    return wrapped_lines


def _wrap_text(text: str, max_line_width: int, max_lines: int) -> str:
    if max_lines == 1 or _display_width(text) <= max_line_width:
        return text

    existing_lines = [part for part in text.replace("\n", r"\N").split(r"\N") if part]
    if len(existing_lines) >= max_lines:
        return r"\N".join(existing_lines[:max_lines])

    split_index = _find_best_split_index(text, max_line_width)
    if split_index <= 0 or split_index >= len(text):
        return text

    left = text[:split_index].rstrip()
    right = text[split_index:].lstrip()
    if not left or not right:
        return text

    return rf"{left}\N{right}"


def _find_best_split_index(text: str, max_line_width: int) -> int:
    total_width = _display_width(text)
    target_width = max(total_width // 2, 1)

    fallback_index = -1
    fallback_score: tuple[int, int] | None = None
    preferred_index = -1
    preferred_score: tuple[int, int] | None = None

    running_width = 0
    for idx, char in enumerate(text[:-1], start=1):
        running_width += _char_width(char)
        left_width = running_width
        right_width = total_width - left_width
        score = (
            abs(left_width - target_width),
            max(left_width, right_width),
        )

        if fallback_score is None or score < fallback_score:
            fallback_score = score
            fallback_index = idx

        if char in PREFERRED_BREAK_CHARS:
            if (
                preferred_score is None
                or score < preferred_score
                or (
                    score == preferred_score
                    and max(left_width, right_width) <= max_line_width
                )
            ):
                preferred_score = score
                preferred_index = idx

    if preferred_index != -1:
        return preferred_index
    return fallback_index


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in text)


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    if unicodedata.east_asian_width(char) in {"F", "W"}:
        return 2
    return 1
