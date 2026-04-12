"""Tests for the corners format extension."""

from autosub.core.schemas import SubtitleLine
from autosub.extensions.corners.main import (
    apply_corners,
    detect_by_cues,
    dedup_consecutive,
    _merge_detections,
)


def _line(text, start=0.0, end=1.0):
    return SubtitleLine(text=text, start_time=start, end_time=end)


SEGMENTS = [
    {"name": "Fan Letter", "description": "Reading fan mail", "cues": ["お便り"]},
    {"name": "Song Corner", "description": "Song segment", "cues": ["曲のコーナー"]},
]


# --- detect_by_cues ---


def testdetect_by_cues_basic():
    lines = [
        _line("今日もよろしく"),
        _line("お便りをいただきました"),
        _line("ありがとう"),
        _line("曲のコーナーです"),
    ]
    result = detect_by_cues(lines, SEGMENTS)
    assert result == [None, "Fan Letter", None, "Song Corner"]


def testdetect_by_cues_no_cues():
    lines = [_line("普通の文")]
    result = detect_by_cues(lines, [{"name": "X", "description": "No cues"}])
    assert result == [None]


def testdetect_by_cues_empty_segments():
    lines = [_line("text")]
    result = detect_by_cues(lines, [])
    assert result == [None]


def testdetect_by_cues_empty_lines():
    result = detect_by_cues([], SEGMENTS)
    assert result == []


# --- dedup_consecutive ---


def testdedup_consecutive_basic():
    corners = [None, "A", "A", None, "B", "B", "A"]
    assert dedup_consecutive(corners) == [None, "A", None, None, "B", None, "A"]


def testdedup_consecutive_all_none():
    assert dedup_consecutive([None, None]) == [None, None]


def testdedup_consecutive_no_dupes():
    assert dedup_consecutive(["A", "B", "A"]) == ["A", "B", "A"]


def test_dedup_consecutive_same_corner_after_none_gap():
    """A show that returns to the same corner after a gap should preserve the second occurrence."""
    corners = ["Fan Letter", None, None, "Fan Letter"]
    assert dedup_consecutive(corners) == ["Fan Letter", None, None, "Fan Letter"]


def test_dedup_consecutive_same_corner_no_gap():
    """Truly consecutive same-corner should still be deduped."""
    corners = ["Fan Letter", "Fan Letter", None, "Song"]
    assert dedup_consecutive(corners) == ["Fan Letter", None, None, "Song"]


# --- _merge_detections ---


def test_merge_llm_takes_precedence():
    cue = [None, "Fan Letter", None]
    llm = ["Song Corner", None, None]
    assert _merge_detections(cue, llm) == ["Song Corner", "Fan Letter", None]


def test_merge_cue_fills_gaps():
    cue = [None, "Fan Letter", None]
    llm = [None, None, None]
    assert _merge_detections(cue, llm) == [None, "Fan Letter", None]


# --- apply_corners ---


def test_apply_corners_cue_engine():
    lines = [
        _line("始まり"),
        _line("お便りをいただきました"),
        _line("続きます"),
        _line("曲のコーナーです"),
    ]
    config = {"segments": SEGMENTS, "engine": "cues"}
    result = apply_corners(lines, config)
    assert result[0].corner is None
    assert result[1].corner == "Fan Letter"
    assert result[2].corner is None
    assert result[3].corner == "Song Corner"


def test_apply_corners_preserves_text_and_timing():
    lines = [_line("テスト", start=1.0, end=2.0)]
    config = {"segments": SEGMENTS, "engine": "cues"}
    result = apply_corners(lines, config)
    assert result[0].text == "テスト"
    assert result[0].start_time == 1.0
    assert result[0].end_time == 2.0


def test_apply_corners_preserves_role():
    line = SubtitleLine(
        text="お便り", start_time=0.0, end_time=1.0, role="host"
    )
    config = {"segments": SEGMENTS, "engine": "cues"}
    result = apply_corners([line], config)
    assert result[0].role == "host"
    assert result[0].corner == "Fan Letter"


def test_apply_cornersdedup_consecutive():
    lines = [
        _line("お便りをいただきました"),
        _line("お便りの続き"),  # Same corner detected again
    ]
    config = {"segments": SEGMENTS, "engine": "cues"}
    result = apply_corners(lines, config)
    assert result[0].corner == "Fan Letter"
    assert result[1].corner is None  # Deduped


def test_apply_corners_empty_lines():
    result = apply_corners([], {"segments": SEGMENTS})
    assert result == []


def test_apply_corners_no_segments():
    lines = [_line("text")]
    result = apply_corners(lines, {"segments": []})
    assert result[0].corner is None


def test_apply_corners_none_config():
    lines = [_line("text")]
    result = apply_corners(lines, None)
    assert result[0].corner is None
