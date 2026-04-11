"""Tests for corners integration in the format pipeline (generator + boundary extraction)."""

import pyass

from autosub.core.schemas import SubtitleLine
from autosub.pipeline.format.generator import generate_ass_file
from autosub.pipeline.translate.main import _extract_corner_boundaries


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


# --- _extract_corner_boundaries ---


def _dialogue(text, start=0.0):
    return pyass.Event(
        format=pyass.EventFormat.DIALOGUE,
        start=pyass.timedelta(seconds=start),
        end=pyass.timedelta(seconds=start + 1.0),
        text=text,
    )


def _corner_comment(name, start=0.0):
    return pyass.Event(
        format=pyass.EventFormat.COMMENT,
        start=pyass.timedelta(seconds=start),
        end=pyass.timedelta(seconds=start + 1.0),
        effect="corner",
        text=f"=== Corner: {name} ===",
    )


def _regular_comment(text, start=0.0):
    return pyass.Event(
        format=pyass.EventFormat.COMMENT,
        start=pyass.timedelta(seconds=start),
        end=pyass.timedelta(seconds=start + 1.0),
        text=text,
    )


def test_extract_boundaries_basic():
    d0 = _dialogue("line 0", start=0.0)
    d1 = _dialogue("line 1", start=1.0)
    d2 = _dialogue("line 2", start=2.0)
    d3 = _dialogue("line 3", start=3.0)

    all_events = [
        d0,
        _corner_comment("Fan Letter", start=1.0),
        d1,
        d2,
        _corner_comment("Song Corner", start=3.0),
        d3,
    ]
    events_to_translate = [d0, d1, d2, d3]

    boundaries = _extract_corner_boundaries(all_events, events_to_translate)
    assert boundaries == [1, 3]


def test_extract_boundaries_no_corners():
    d0 = _dialogue("line 0")
    d1 = _dialogue("line 1")
    all_events = [d0, d1]
    boundaries = _extract_corner_boundaries(all_events, [d0, d1])
    assert boundaries == []


def test_extract_boundaries_ignores_regular_comments():
    d0 = _dialogue("line 0")
    d1 = _dialogue("line 1")
    all_events = [d0, _regular_comment("just a note"), d1]
    boundaries = _extract_corner_boundaries(all_events, [d0, d1])
    assert boundaries == []


def test_extract_boundaries_corner_at_start():
    d0 = _dialogue("line 0")
    d1 = _dialogue("line 1")
    all_events = [_corner_comment("Opening"), d0, d1]
    boundaries = _extract_corner_boundaries(all_events, [d0, d1])
    assert boundaries == [0]


def test_extract_boundaries_consecutive_corners():
    """Two corner comments before same dialogue line — both map to same index."""
    d0 = _dialogue("line 0")
    d1 = _dialogue("line 1")
    all_events = [
        d0,
        _corner_comment("A"),
        _corner_comment("B"),
        d1,
    ]
    # Both corners point to dialogue index 1, but only the last pending is captured
    boundaries = _extract_corner_boundaries(all_events, [d0, d1])
    assert 1 in boundaries
