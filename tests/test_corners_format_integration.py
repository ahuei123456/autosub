"""Tests for corners integration in the format pipeline (generator + boundary extraction)."""

import pyass

from autosub.core.schemas import SubtitleCue, SubtitleDocument, SubtitleLine
from autosub.pipeline.format.generator import generate_ass_file
from autosub.pipeline.translate.main import _extract_corner_boundaries_from_cues


def _line(text, start=0.0, end=1.0, corner=None, role=None):
    return SubtitleLine(
        text=text, start_time=start, end_time=end, corner=corner, role=role
    )


# --- generator: corner Comment events ---


def test_generator_emits_corner_comment(tmp_path):
    lines = [
        _line("始まり", start=0.0, end=1.0),
        _line("お便りコーナー", start=1.0, end=2.0, corner="Fan Letter"),
        _line("内容です", start=2.0, end=3.0),
    ]
    out = tmp_path / "test.ass"
    generate_ass_file(lines, out)

    with open(out, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    # Should have 4 events: dialogue, corner comment, dialogue, dialogue
    assert len(script.events) == 4

    # First event is dialogue
    assert script.events[0].format == pyass.EventFormat.DIALOGUE
    assert script.events[0].text == "始まり"

    # Second event is corner comment
    assert script.events[1].format == pyass.EventFormat.COMMENT
    assert script.events[1].effect == "corner"
    assert "Fan Letter" in script.events[1].text

    # Third event is the dialogue line that triggered the corner
    assert script.events[2].format == pyass.EventFormat.DIALOGUE
    assert script.events[2].text == "お便りコーナー"

    # Fourth event
    assert script.events[3].format == pyass.EventFormat.DIALOGUE


def test_generator_no_corner_no_comment(tmp_path):
    lines = [_line("plain text")]
    out = tmp_path / "test.ass"
    generate_ass_file(lines, out)

    with open(out, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    assert len(script.events) == 1
    assert script.events[0].format == pyass.EventFormat.DIALOGUE


def test_generator_multiple_corners(tmp_path):
    lines = [
        _line("開始", corner="Opening"),
        _line("中間", corner="Middle"),
        _line("終了", corner="Ending"),
    ]
    out = tmp_path / "test.ass"
    generate_ass_file(lines, out)

    with open(out, "r", encoding="utf-8") as f:
        script = pyass.load(f)

    comments = [e for e in script.events if e.format == pyass.EventFormat.COMMENT]
    assert len(comments) == 3
    assert "Opening" in comments[0].text
    assert "Middle" in comments[1].text
    assert "Ending" in comments[2].text


# --- _extract_corner_boundaries_from_cues ---


def _cue(text, index, corner=None):
    return SubtitleCue(
        id=f"cue-{index:05d}",
        start_time=float(index),
        end_time=float(index + 1),
        source_text=text,
        corner=corner,
    )


def test_extract_boundaries_basic():
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            _cue("line 0", 0),
            _cue("line 1", 1, corner="Fan Letter"),
            _cue("line 2", 2),
            _cue("line 3", 3, corner="Song Corner"),
        ],
    )

    boundaries = _extract_corner_boundaries_from_cues(document)
    assert boundaries == [1, 3]


def test_extract_boundaries_no_corners():
    document = SubtitleDocument(
        stage="formatted", cues=[_cue("line 0", 0), _cue("line 1", 1)]
    )

    boundaries = _extract_corner_boundaries_from_cues(document)
    assert boundaries == []


def test_extract_boundaries_skips_empty_source_text():
    document = SubtitleDocument(
        stage="formatted",
        cues=[_cue("line 0", 0), _cue("", 1, corner="Ignored"), _cue("line 1", 2)],
    )

    boundaries = _extract_corner_boundaries_from_cues(document)
    assert boundaries == []


def test_extract_boundaries_corner_at_start():
    document = SubtitleDocument(
        stage="formatted", cues=[_cue("line 0", 0, corner="Opening"), _cue("line 1", 1)]
    )

    boundaries = _extract_corner_boundaries_from_cues(document)
    assert boundaries == [0]


def test_extract_boundaries_back_to_back_corners():
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            _cue("line 0", 0),
            _cue("line 1", 1, corner="A"),
            _cue("line 2", 2, corner="B"),
        ],
    )

    boundaries = _extract_corner_boundaries_from_cues(document)
    assert boundaries == [1, 2]
