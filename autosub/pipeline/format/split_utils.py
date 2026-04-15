from __future__ import annotations

from autosub.core.schemas import ReplacementSpan, SubtitleLine, TranscribedWord


def find_split_time(line: SubtitleLine, split_char_pos: int) -> float:
    """
    Given a character position in line.text (post-replacement), return the
    audio timestamp at which to split.

    Resolution order:
    1. If split_char_pos falls inside a replacement span, snap to the end of the
       original source text (span.orig_end) — avoids character-count mismatch
       caused by replacements that change string length.
    2. Otherwise, adjust split_char_pos for the cumulative length delta of all
       replacement spans that ended before the split point, converting from
       replaced-text coordinates to original word-text coordinates.
    3. Walk line.words accumulating character lengths until the running total
       meets or exceeds orig_pos; return that word's end_time.
    4. Falls back to proportional estimation when line.words is empty.
    """
    if not line.words:
        ratio = split_char_pos / max(len(line.text), 1)
        return line.start_time + (line.end_time - line.start_time) * ratio

    # Case 1: split lands inside a replacement span → snap to orig_end
    for span in line.replacement_spans:
        if span.replaced_start <= split_char_pos < span.replaced_end:
            orig_pos = span.orig_end
            break
    else:
        # Case 2: adjust for all replacements that ended before the split point
        offset = sum(
            (span.replaced_end - span.replaced_start)
            - (span.orig_end - span.orig_start)
            for span in line.replacement_spans
            if span.replaced_end <= split_char_pos
        )
        orig_pos = split_char_pos - offset

    # Case 3: walk words to find which one covers orig_pos
    accumulated = 0
    for word in line.words:
        accumulated += len(word.word)
        if accumulated >= orig_pos:
            return word.end_time
    return line.words[-1].end_time


def partition_words(
    words: list[TranscribedWord], split_time: float
) -> tuple[list[TranscribedWord], list[TranscribedWord]]:
    """Split words into those ending at or before split_time and those after."""
    first = [w for w in words if w.end_time <= split_time]
    second = [w for w in words if w.end_time > split_time]
    return first, second


def partition_spans(
    spans: list[ReplacementSpan], split_char_pos: int
) -> tuple[list[ReplacementSpan], list[ReplacementSpan]]:
    """
    Split replacement spans at a character position in replaced text.

    First list:  spans fully before the split (replaced_end <= split_char_pos).
    Second list: spans fully after the split (replaced_start >= split_char_pos),
                 with both replaced and orig coordinates adjusted to be relative
                 to the start of the new sub-line.
    Spans straddling the split are dropped from both lists — this should not
    occur in practice because find_split_time always snaps to word boundaries
    that align with span edges.
    """
    # Compute orig offset for the second sub-line: the position in the original
    # word text that corresponds to split_char_pos in the replaced text.
    orig_offset = split_char_pos - sum(
        (span.replaced_end - span.replaced_start) - (span.orig_end - span.orig_start)
        for span in spans
        if span.replaced_end <= split_char_pos
    )

    first: list[ReplacementSpan] = []
    second: list[ReplacementSpan] = []
    for span in spans:
        if span.replaced_end <= split_char_pos:
            first.append(span)
        elif span.replaced_start >= split_char_pos:
            second.append(
                ReplacementSpan(
                    orig_start=span.orig_start - orig_offset,
                    orig_end=span.orig_end - orig_offset,
                    replaced_start=span.replaced_start - split_char_pos,
                    replaced_end=span.replaced_end - split_char_pos,
                )
            )
        # straddling: drop from both

    return first, second
