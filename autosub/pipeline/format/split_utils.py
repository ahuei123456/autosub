from __future__ import annotations

from autosub.core.schemas import ReplacementSpan, SubtitleLine, TranscribedWord


def find_split_time(line: SubtitleLine, split_char_pos: int) -> float:
    """
    Given a character position in line.text (post-replacement), return the
    audio timestamp at which to split.

    Resolution order:
    1. If no replacement spans exist and line.words already concatenate to
       line.text, walk those words directly and return the boundary word's
       end_time.
    2. If replacement spans exist and line.words are still in original-text
       coordinates, map split_char_pos back through replacement spans to an
       original-text character position, then resolve that character position
       against the original word ranges.
    3. If replacement spans exist but line.words have already been normalized to
       line.text, resolve split_char_pos directly against normalized word ranges,
       interpolating inside merged replacement words instead of snapping to the
       merged word's end.
    4. Falls back to proportional estimation when line.words is empty.
    """
    if not line.words:
        ratio = split_char_pos / max(len(line.text), 1)
        return line.start_time + (line.end_time - line.start_time) * ratio

    word_text = "".join(word.word for word in line.words)
    if not line.replacement_spans and word_text == line.text:
        accumulated = 0
        for word in line.words:
            accumulated += len(word.word)
            if accumulated >= split_char_pos:
                return word.end_time
        return line.words[-1].end_time

    if line.replacement_spans and word_text == line.text:
        return _time_at_char_position(line.words, float(split_char_pos))

    orig_pos = _replaced_to_original_char_position(
        line.replacement_spans, float(split_char_pos)
    )
    return _time_at_char_position(line.words, orig_pos)


def _replaced_to_original_char_position(
    spans: list[ReplacementSpan], split_char_pos: float
) -> float:
    for span in spans:
        if span.replaced_start <= split_char_pos <= span.replaced_end:
            replaced_len = span.replaced_end - span.replaced_start
            if replaced_len <= 0:
                return float(span.orig_end)
            ratio = (split_char_pos - span.replaced_start) / replaced_len
            orig_len = span.orig_end - span.orig_start
            return span.orig_start + (orig_len * ratio)

    offset = sum(
        (span.replaced_end - span.replaced_start) - (span.orig_end - span.orig_start)
        for span in spans
        if span.replaced_end <= split_char_pos
    )
    return split_char_pos - offset


def _time_at_char_position(words: list[TranscribedWord], char_pos: float) -> float:
    if not words:
        return 0.0
    if char_pos <= 0:
        return words[0].start_time

    accumulated = 0.0
    for word in words:
        word_len = len(word.word)
        next_accumulated = accumulated + word_len
        if char_pos <= next_accumulated:
            return _interpolate_word_time(word, char_pos - accumulated)
        accumulated = next_accumulated

    return words[-1].end_time


def _interpolate_word_time(word: TranscribedWord, char_offset: float) -> float:
    word_len = len(word.word)
    if word_len <= 0:
        return word.start_time
    if char_offset <= 0:
        return word.start_time
    if char_offset >= word_len:
        return word.end_time
    ratio = char_offset / word_len
    return word.start_time + ((word.end_time - word.start_time) * ratio)


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
