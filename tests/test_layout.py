from autosub.core.schemas import SubtitleLine
from autosub.pipeline.format.layout import wrap_subtitle_lines


def test_wrap_subtitle_lines_inserts_single_ass_break():
    lines = [
        SubtitleLine(
            text="今日は本当にありがとうございましたまたお会いしましょう",
            start_time=0.0,
            end_time=2.0,
            speaker=None,
        )
    ]

    result = wrap_subtitle_lines(lines, max_line_width=18, max_lines_per_subtitle=2)

    assert len(result) == 1
    assert result[0].text.count(r"\N") == 1


def test_wrap_subtitle_lines_prefers_punctuation_breaks():
    lines = [
        SubtitleLine(
            text="今日は、みなさん本当にありがとうございました",
            start_time=0.0,
            end_time=2.0,
            speaker=None,
        )
    ]

    result = wrap_subtitle_lines(lines, max_line_width=16, max_lines_per_subtitle=2)

    assert result[0].text.startswith("今日は、")
    assert r"\N" in result[0].text


def test_wrap_subtitle_lines_keeps_short_text_unchanged():
    lines = [SubtitleLine(text="ありがとう。", start_time=0.0, end_time=1.0)]

    result = wrap_subtitle_lines(lines, max_line_width=22, max_lines_per_subtitle=2)

    assert result[0].text == "ありがとう。"
