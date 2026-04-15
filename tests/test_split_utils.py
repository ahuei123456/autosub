"""
Tests for autosub.pipeline.format.split_utils.

Real word fixtures are extracted from nonshichotto/144/output.json:
  - TOITADAKIMASHITA_WORDS: words covering the といただきました suffix (verbatim, no replacement)
  - NONBANWA_REPLACEMENT_WORDS: words where の番は was replaced with のんばんは
  - NONBANWA_VERBATIM_WORDS: words where のんばんは appears verbatim in the transcript
"""

import pytest

from autosub.core.schemas import ReplacementSpan, SubtitleLine, TranscribedWord
from autosub.pipeline.format.split_utils import (
    find_split_time,
    partition_spans,
    partition_words,
)

# ---------------------------------------------------------------------------
# Real word fixtures (extracted from nonshichotto/144/output.json)
# ---------------------------------------------------------------------------

# Words covering "夢の国行けるといいですねといただきました。"
# といただきました starts at char 12 in the concatenated word text.
TOITADAKIMASHITA_WORDS = [
    TranscribedWord(word="夢", start_time=587.82, end_time=588.1),
    TranscribedWord(word="の", start_time=588.1, end_time=588.22),
    TranscribedWord(word="国", start_time=588.22, end_time=588.5),
    TranscribedWord(word="行ける", start_time=588.5, end_time=588.86),
    TranscribedWord(word="と", start_time=588.86, end_time=589.02),
    TranscribedWord(word="いい", start_time=589.02, end_time=589.14),
    TranscribedWord(word="です", start_time=589.14, end_time=589.46),
    TranscribedWord(word="ね", start_time=589.46, end_time=589.86),
    TranscribedWord(word="と", start_time=589.86, end_time=589.94),
    TranscribedWord(word="いただき", start_time=589.94, end_time=590.42),
    TranscribedWord(word="まし", start_time=590.42, end_time=590.7),
    TranscribedWord(word="た。", start_time=590.7, end_time=590.86),
]
# Concatenated: 夢(1)の(1)国(1)行ける(3)と(1)いい(2)です(2)ね(1)と(1)いただき(4)まし(2)た。(2) = 21 chars
# といただきました。 = 9 chars → split_char_pos = 21 - 9 = 12
# Word at char 12: accumulated after ね = 1+1+1+3+1+2+2+1 = 12 → ね.end_time = 589.86

# Words where の番は (3 chars) was replaced with のんばんは (5 chars).
# Original word text: のちゃん、の番は何番は?  (12 chars)
# Replaced text:      のちゃん、のんばんは何番は? (14 chars)
# ReplacementSpan: orig=[5,8), replaced=[5,10)
NONBANWA_REPLACEMENT_WORDS = [
    TranscribedWord(word="の", start_time=567.54, end_time=567.62),
    TranscribedWord(word="ちゃん、", start_time=567.62, end_time=567.94),
    TranscribedWord(word="の", start_time=567.94, end_time=568.1),
    TranscribedWord(word="番", start_time=568.1, end_time=568.34),
    TranscribedWord(word="は", start_time=568.34, end_time=568.54),
    TranscribedWord(word="何", start_time=568.54, end_time=568.82),
    TranscribedWord(word="番", start_time=568.82, end_time=569.1),
    TranscribedWord(word="は?", start_time=569.1, end_time=569.46),
]
NONBANWA_REPLACEMENT_SPAN = ReplacementSpan(
    orig_start=5, orig_end=8, replaced_start=5, replaced_end=10
)

# Words where のんばんは appears verbatim (は。 is fused as one word).
# Original word text: のんばんは。先日農市
# のんばんは is 5 chars; split_char_pos = 5
# Word at char 5: accumulated after のん(2)+ばん(2)+は。(2) — は。 covers chars 4-6, so
#   acc after ばん = 4 < 5, acc after は。 = 6 >= 5 → は。.end_time = 709.8
NONBANWA_VERBATIM_WORDS = [
    TranscribedWord(word="のん", start_time=709.16, end_time=709.36),
    TranscribedWord(word="ばん", start_time=709.36, end_time=709.64),
    TranscribedWord(word="は。", start_time=709.64, end_time=709.8),
    TranscribedWord(word="先日", start_time=711.24, end_time=711.88),
    TranscribedWord(word="農市", start_time=712.0, end_time=712.32),
]


# ---------------------------------------------------------------------------
# find_split_time tests
# ---------------------------------------------------------------------------


def test_find_split_time_no_words_falls_back_to_proportional():
    line = SubtitleLine(text="abcde", start_time=0.0, end_time=10.0)
    # split at char 2 out of 5 → 40% → 4.0s
    assert find_split_time(line, 2) == pytest.approx(4.0)


def test_find_split_time_verbatim_suffix_uses_word_boundary():
    """といただきました appears verbatim; split should snap to ね.end_time."""
    line = SubtitleLine(
        text="夢の国行けるといいですねといただきました。",
        start_time=587.82,
        end_time=590.86,
        words=TOITADAKIMASHITA_WORDS,
    )
    # split_char_pos = 12 (end of main body, start of suffix)
    result = find_split_time(line, 12)
    assert result == pytest.approx(589.86)


def test_find_split_time_at_span_boundary_uses_last_replaced_word():
    """Split at the end of a replacement span → resolves via offset to は.end_time."""
    line = SubtitleLine(
        text="のちゃん、のんばんは何番は?",
        start_time=567.54,
        end_time=569.46,
        words=NONBANWA_REPLACEMENT_WORDS,
        replacement_spans=[NONBANWA_REPLACEMENT_SPAN],
    )
    # split_char_pos = 10 (replaced_end of the span, end of のんばんは)
    # span.replaced_end=10 <= 10 → offset branch: orig_pos = 10 - (5-3) = 8
    # Word at orig_pos 8: は (chars 7-8) → は.end_time = 568.54
    result = find_split_time(line, 10)
    assert result == pytest.approx(568.54)


def test_find_split_time_inside_replacement_span_snaps_to_orig_end():
    """Split inside the span (e.g. char 7, middle of のんばんは) → same result."""
    line = SubtitleLine(
        text="のちゃん、のんばんは何番は?",
        start_time=567.54,
        end_time=569.46,
        words=NONBANWA_REPLACEMENT_WORDS,
        replacement_spans=[NONBANWA_REPLACEMENT_SPAN],
    )
    # span covers replaced [5,10); split at 7 is inside → orig_pos = span.orig_end = 8
    result = find_split_time(line, 7)
    assert result == pytest.approx(568.54)


def test_find_split_time_verbatim_nonbanwa_fused_punctuation():
    """のんばんは verbatim where は。 is a single word token."""
    line = SubtitleLine(
        text="のんばんは。先日農市",
        start_time=709.16,
        end_time=712.32,
        words=NONBANWA_VERBATIM_WORDS,
    )
    # split_char_pos = 5 (end of のんばんは); は。 covers chars 4-6 → は。.end_time = 709.8
    result = find_split_time(line, 5)
    assert result == pytest.approx(709.8)


def test_find_split_time_after_replacement_adjusts_offset():
    """Split after a preceding replacement correctly shifts the orig position."""
    # text = "XYZのんばんはABC" where "XY" (2 chars) was replaced by "XYZ" (3 chars)
    # span: orig=[0,2), replaced=[0,3)
    # word text: "XYのんばんはABC"
    words = [
        TranscribedWord(word="XY", start_time=0.0, end_time=1.0),
        TranscribedWord(word="のんばんは", start_time=1.0, end_time=2.0),
        TranscribedWord(word="ABC", start_time=2.0, end_time=3.0),
    ]
    span = ReplacementSpan(orig_start=0, orig_end=2, replaced_start=0, replaced_end=3)
    line = SubtitleLine(
        text="XYZのんばんはABC",
        start_time=0.0,
        end_time=3.0,
        words=words,
        replacement_spans=[span],
    )
    # split after "のんばんは" in replaced text → char 3+5 = 8
    # span.replaced_end=3 <= 8 → offset = (3-0)-(2-0) = 1; orig_pos = 8-1 = 7
    # walk words: XY=2, のんばんは=7 (acc=2+5=7 >= 7) → のんばんは.end_time = 2.0
    result = find_split_time(line, 8)
    assert result == pytest.approx(2.0)


def test_find_split_time_no_words_zero_duration():
    line = SubtitleLine(text="hello", start_time=5.0, end_time=5.0)
    result = find_split_time(line, 3)
    assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# partition_words tests
# ---------------------------------------------------------------------------


def test_partition_words_splits_at_boundary():
    words = [
        TranscribedWord(word="a", start_time=0.0, end_time=1.0),
        TranscribedWord(word="b", start_time=1.0, end_time=2.0),
        TranscribedWord(word="c", start_time=2.0, end_time=3.0),
    ]
    first, second = partition_words(words, 2.0)
    assert [w.word for w in first] == ["a", "b"]
    assert [w.word for w in second] == ["c"]


def test_partition_words_all_first():
    words = [
        TranscribedWord(word="a", start_time=0.0, end_time=1.0),
    ]
    first, second = partition_words(words, 5.0)
    assert len(first) == 1
    assert len(second) == 0


def test_partition_words_all_second():
    words = [
        TranscribedWord(word="a", start_time=0.0, end_time=1.0),
    ]
    first, second = partition_words(words, 0.0)
    assert len(first) == 0
    assert len(second) == 1


# ---------------------------------------------------------------------------
# partition_spans tests
# ---------------------------------------------------------------------------


def test_partition_spans_fully_before():
    span = ReplacementSpan(orig_start=0, orig_end=3, replaced_start=0, replaced_end=5)
    first, second = partition_spans([span], 10)
    assert first == [span]
    assert second == []


def test_partition_spans_fully_after_adjusts_coords():
    span = ReplacementSpan(orig_start=5, orig_end=8, replaced_start=8, replaced_end=13)
    first, second = partition_spans([span], 5)
    assert first == []
    assert len(second) == 1
    s = second[0]
    # replaced coords: 8-5=3, 13-5=8
    assert s.replaced_start == 3
    assert s.replaced_end == 8
    # orig offset: split at replaced=5, no spans before → orig_offset = 5
    # orig coords: 5-5=0, 8-5=3
    assert s.orig_start == 0
    assert s.orig_end == 3


def test_partition_spans_at_exact_boundary_goes_to_first():
    # span.replaced_end == split_char_pos → goes to first list
    span = ReplacementSpan(orig_start=0, orig_end=3, replaced_start=0, replaced_end=5)
    first, second = partition_spans([span], 5)
    assert first == [span]
    assert second == []


def test_partition_spans_multiple_spans():
    span1 = ReplacementSpan(orig_start=0, orig_end=3, replaced_start=0, replaced_end=5)
    span2 = ReplacementSpan(orig_start=6, orig_end=9, replaced_start=8, replaced_end=13)
    first, second = partition_spans([span1, span2], 6)
    assert first == [span1]
    assert len(second) == 1
    s = second[0]
    # orig_offset = 6 - (5-3) = 4; replaced coords: 8-6=2, 13-6=7; orig: 6-4=2, 9-4=5
    assert s.replaced_start == 2
    assert s.replaced_end == 7
    assert s.orig_start == 2
    assert s.orig_end == 5


def test_partition_spans_empty():
    first, second = partition_spans([], 5)
    assert first == []
    assert second == []
