"""
Tests for the split_after feature: _apply_replacements_with_spans and apply_split_after.

Real word fixtures are extracted from nonshichotto/144/output.json.
"""

import json

import pyass
import pytest

from autosub.core.schemas import ReplacementSpan, SubtitleLine, TranscribedWord
from autosub.pipeline.format.main import (
    _apply_replacements_with_spans,
    apply_split_after,
    format_subtitles,
)

# ---------------------------------------------------------------------------
# Real word fixtures (see test_split_utils.py for full context)
# ---------------------------------------------------------------------------

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

NONBANWA_VERBATIM_WORDS = [
    TranscribedWord(word="のん", start_time=709.16, end_time=709.36),
    TranscribedWord(word="ばん", start_time=709.36, end_time=709.64),
    TranscribedWord(word="は。", start_time=709.64, end_time=709.8),
    TranscribedWord(word="先日", start_time=711.24, end_time=711.88),
    TranscribedWord(word="農市", start_time=712.0, end_time=712.32),
]


# ---------------------------------------------------------------------------
# _apply_replacements_with_spans tests
# ---------------------------------------------------------------------------


def test_apply_replacements_no_match():
    text, spans = _apply_replacements_with_spans("こんにちは", {"の番は": "のんばんは"})
    assert text == "こんにちは"
    assert spans == []


def test_apply_replacements_single_match():
    text, spans = _apply_replacements_with_spans("の番は", {"の番は": "のんばんは"})
    assert text == "のんばんは"
    assert len(spans) == 1
    assert spans[0] == ReplacementSpan(
        orig_start=0, orig_end=3, replaced_start=0, replaced_end=5
    )


def test_apply_replacements_match_in_middle():
    text, spans = _apply_replacements_with_spans(
        "のちゃん、の番は何番は?", {"の番は": "のんばんは"}
    )
    assert text == "のちゃん、のんばんは何番は?"
    assert len(spans) == 1
    assert spans[0] == ReplacementSpan(
        orig_start=5, orig_end=8, replaced_start=5, replaced_end=10
    )


def test_apply_replacements_multiple_sources_same_target():
    text, spans = _apply_replacements_with_spans(
        "のん番は",
        {"の番は": "のんばんは", "のん番は": "のんばんは"},
    )
    # "のん番は" (4 chars) is longer than "の番は" (3 chars) and matches first
    assert text == "のんばんは"
    assert len(spans) == 1
    assert spans[0].orig_end == 4


def test_apply_replacements_no_overlap():
    # Two replacements at non-overlapping positions
    text, spans = _apply_replacements_with_spans(
        "ABfooCD bar EF",
        {"foo": "FOO", "bar": "BAR"},
    )
    assert text == "ABFOOCD BAR EF"
    assert len(spans) == 2
    assert spans[0] == ReplacementSpan(
        orig_start=2, orig_end=5, replaced_start=2, replaced_end=5
    )
    assert spans[1] == ReplacementSpan(
        orig_start=8, orig_end=11, replaced_start=8, replaced_end=11
    )


def test_apply_replacements_length_change_updates_replaced_coords():
    # "ab" (2) → "xyz" (3): replaced coords shift by +1 for subsequent text
    text, spans = _apply_replacements_with_spans("ab cd", {"ab": "xyz"})
    assert text == "xyz cd"
    assert spans[0].replaced_start == 0
    assert spans[0].replaced_end == 3


# ---------------------------------------------------------------------------
# apply_split_after tests
# ---------------------------------------------------------------------------


def test_apply_split_after_no_match_returns_line_unchanged():
    line = SubtitleLine(text="こんにちは今日もよろしく", start_time=0.0, end_time=5.0)
    result = apply_split_after([line], ["のんばんは"])
    assert len(result) == 1
    assert result[0].text == "こんにちは今日もよろしく"


def test_apply_split_after_phrase_at_end_not_split():
    # phrase ending exactly at end of text → no split
    line = SubtitleLine(text="こんにちはのんばんは", start_time=0.0, end_time=5.0)
    result = apply_split_after([line], ["のんばんは"])
    assert len(result) == 1


def test_apply_split_after_verbatim_phrase_uses_word_timestamp():
    """のんばんは verbatim — split time should use は。.end_time = 709.8."""
    line = SubtitleLine(
        text="のんばんは。先日農市",
        start_time=709.16,
        end_time=712.32,
        words=NONBANWA_VERBATIM_WORDS,
    )
    result = apply_split_after([line], ["のんばんは"])
    assert len(result) == 2
    assert result[0].text == "のんばんは"
    assert result[1].text == "。先日農市"
    assert result[0].end_time == pytest.approx(709.8)
    assert result[1].start_time == pytest.approx(709.8)


def test_apply_split_after_replacement_phrase_uses_word_timestamp():
    """の番は was replaced with のんばんは — split time should use は.end_time = 568.54."""
    line = SubtitleLine(
        text="のちゃん、のんばんは何番は?",
        start_time=567.54,
        end_time=569.46,
        words=NONBANWA_REPLACEMENT_WORDS,
        replacement_spans=[
            ReplacementSpan(orig_start=5, orig_end=8, replaced_start=5, replaced_end=10)
        ],
    )
    result = apply_split_after([line], ["のんばんは"])
    assert len(result) == 2
    assert result[0].text == "のちゃん、のんばんは"
    assert result[1].text == "何番は?"
    assert result[0].end_time == pytest.approx(568.54)
    assert result[1].start_time == pytest.approx(568.54)


def test_apply_split_after_multiple_occurrences_in_one_line():
    line = SubtitleLine(
        text="aXbc Xde",
        start_time=0.0,
        end_time=8.0,
        words=[
            TranscribedWord(word="aX", start_time=0.0, end_time=2.0),
            TranscribedWord(word="bc ", start_time=2.0, end_time=4.0),
            TranscribedWord(word="Xde", start_time=4.0, end_time=8.0),
        ],
    )
    result = apply_split_after([line], ["X"])
    assert len(result) == 3
    assert result[0].text == "aX"
    assert result[1].text == "bc X"
    assert result[2].text == "de"


def test_apply_split_after_words_partitioned_correctly():
    """Words are distributed to the correct sub-lines."""
    line = SubtitleLine(
        text="のちゃん、のんばんは何番は?",
        start_time=567.54,
        end_time=569.46,
        words=NONBANWA_REPLACEMENT_WORDS,
        replacement_spans=[
            ReplacementSpan(orig_start=5, orig_end=8, replaced_start=5, replaced_end=10)
        ],
    )
    result = apply_split_after([line], ["のんばんは"])
    # split_time = 568.54 (は.end_time)
    # first line words: end_time <= 568.54
    first_words = [w.word for w in result[0].words]
    assert first_words == ["の", "ちゃん、", "の", "番", "は"]
    # second line words: end_time > 568.54
    second_words = [w.word for w in result[1].words]
    assert second_words == ["何", "番", "は?"]


def test_apply_split_after_multiple_lines():
    lines = [
        SubtitleLine(text="のんばんは今日もよろしく", start_time=0.0, end_time=5.0),
        SubtitleLine(text="変わらない", start_time=5.0, end_time=8.0),
        SubtitleLine(text="のんばんはまた明日", start_time=8.0, end_time=12.0),
    ]
    result = apply_split_after(lines, ["のんばんは"])
    assert len(result) == 5
    assert result[0].text == "のんばんは"
    assert result[1].text == "今日もよろしく"
    assert result[2].text == "変わらない"
    assert result[3].text == "のんばんは"
    assert result[4].text == "また明日"


def test_apply_split_after_spans_propagated_to_first_subline():
    """The first sub-line receives the replacement spans that fall within it."""
    line = SubtitleLine(
        text="のちゃん、のんばんは何番は?",
        start_time=567.54,
        end_time=569.46,
        words=NONBANWA_REPLACEMENT_WORDS,
        replacement_spans=[
            ReplacementSpan(orig_start=5, orig_end=8, replaced_start=5, replaced_end=10)
        ],
    )
    result = apply_split_after([line], ["のんばんは"])
    # span (replaced_end=10) is at the boundary of the split (split_char_pos=10)
    # → goes to first sub-line
    assert len(result[0].replacement_spans) == 1
    assert len(result[1].replacement_spans) == 0


# ---------------------------------------------------------------------------
# Integration: format_subtitles with split_after
# ---------------------------------------------------------------------------


def test_format_subtitles_greetings_via_radio_discourse(tmp_path):
    """End-to-end: の番は is replaced with のんばんは, then split after it via radio_discourse greetings."""
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "out.ass"

    words = [
        {"word": "の", "start_time": 0.0, "end_time": 0.1},
        {"word": "ちゃん、", "start_time": 0.1, "end_time": 0.5},
        {"word": "の", "start_time": 0.5, "end_time": 0.6},
        {"word": "番", "start_time": 0.6, "end_time": 0.7},
        {"word": "は", "start_time": 0.7, "end_time": 0.8},
        {"word": "今日も", "start_time": 0.9, "end_time": 1.5},
        {"word": "よろしく。", "start_time": 1.5, "end_time": 2.0},
    ]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8"
    )

    format_subtitles(
        transcript_path,
        output_path,
        replacements={"の番は": "のんばんは"},
        extensions_config={
            "radio_discourse": {"enabled": True, "greetings": ["のんばんは"]}
        },
    )

    with open(output_path, encoding="utf-8") as f:
        script = pyass.load(f)

    events = [e for e in script.events if isinstance(e, pyass.Event)]
    texts = [e.text for e in events]
    assert any("のんばんは" in t for t in texts), f"Expected のんばんは in {texts}"
    assert any("今日も" in t or "よろしく" in t for t in texts), (
        f"Expected continuation line in {texts}"
    )
    nonbanwa_idx = next(i for i, t in enumerate(texts) if "のんばんは" in t)
    if nonbanwa_idx + 1 < len(events):
        assert events[nonbanwa_idx].end <= events[nonbanwa_idx + 1].start
