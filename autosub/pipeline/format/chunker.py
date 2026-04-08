from typing import List
from autosub.core.schemas import TranscribedWord, SubtitleLine

HARD_PUNCTUATION = {"。", "！", "？", ".", "!", "?"}
SOFT_PUNCTUATION = {"、", ","}

LONG_PAUSE_THRESHOLD = 1.5
SOFT_PAUSE_THRESHOLD = 0.6
SOFT_PUNCT_WORD_THRESHOLD = 16


def chunk_words_to_lines(words: List[TranscribedWord]) -> List[SubtitleLine]:
    """
    Groups a continuous stream of TranscribedWord objects into semantic SubtitleLines.
    Splits occur on:
    1. Hard Punctuation (。, !, ?)
    2. Long Pauses (> 1.5s)
    3. Soft Punctuation (、, ,) + Moderate Pause (> 0.6s)
    4. Soft Punctuation after an already-dense chunk (>= 12 transcribed words)

    Words are grouped by speaker before chunking to handle overlapping speech.
    """
    from collections import defaultdict

    speaker_groups: dict[str, List[TranscribedWord]] = defaultdict(list)
    for w in words:
        # Default to "Speaker_1" if no speaker is provided by the API
        speaker_id = w.speaker if w.speaker else "Speaker_1"
        speaker_groups[speaker_id].append(w)

    all_lines: List[SubtitleLine] = []

    for speaker_id, speaker_words in speaker_groups.items():
        current_chunk: List[TranscribedWord] = []

        for i, word in enumerate(speaker_words):
            if word.start_time == word.end_time:
                continue

            current_chunk.append(word)

            # Check if this is the last word for this speaker
            if i == len(speaker_words) - 1:
                all_lines.append(_create_line(current_chunk))
                break

            next_word = speaker_words[i + 1]
            pause_duration = next_word.start_time - word.end_time

            # Rule 1: Hard Punctuation
            is_hard_punct = any(word.word.endswith(p) for p in HARD_PUNCTUATION)

            # Rule 2: Long Pause (trailed off thought)
            is_long_pause = pause_duration >= LONG_PAUSE_THRESHOLD

            # Rule 3: Soft Punctuation + Moderate Pause
            is_soft_punct = any(word.word.endswith(p) for p in SOFT_PUNCTUATION)
            is_soft_pause = is_soft_punct and pause_duration >= SOFT_PAUSE_THRESHOLD
            is_dense_soft_break = (
                is_soft_punct and len(current_chunk) >= SOFT_PUNCT_WORD_THRESHOLD
            )

            if is_hard_punct or is_long_pause or is_soft_pause or is_dense_soft_break:
                all_lines.append(_create_line(current_chunk))
                current_chunk = []

    # Sort all generated lines chronologically by start time
    all_lines.sort(key=lambda line: line.start_time)

    return all_lines


def _create_line(chunk: List[TranscribedWord]) -> SubtitleLine:
    """Helper to merge a list of words into a single SubtitleLine object."""
    if not chunk:
        raise ValueError("Cannot create a SubtitleLine from an empty chunk.")

    text = "".join(w.word for w in chunk)
    start_time = chunk[0].start_time
    end_time = chunk[-1].end_time
    speaker = chunk[
        0
    ].speaker  # Assume speaker is uniform across a single semantic chunk

    return SubtitleLine(
        text=text, start_time=start_time, end_time=end_time, speaker=speaker
    )
