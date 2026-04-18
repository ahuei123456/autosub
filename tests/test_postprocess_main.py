from pathlib import Path

import pyass

from autosub.pipeline.postprocess.main import _ensure_quoted, postprocess_subtitles


def _write_ass(path: Path, events: list[pyass.Event]) -> None:
    script = pyass.Script(styles=[pyass.Style(name="Default")], events=events)
    with open(path, "w", encoding="utf-8") as handle:
        pyass.dump(script, handle)


def test_postprocess_quotes_listener_mail_replace_mode(tmp_path):
    ass_path = tmp_path / "translated.ass"
    _write_ass(
        ass_path,
        [
            pyass.Event(
                start=pyass.timedelta(seconds=0),
                end=pyass.timedelta(seconds=1),
                style="Default",
                name="listener_mail",
                text="This is a listener message.",
            ),
            pyass.Event(
                start=pyass.timedelta(seconds=1),
                end=pyass.timedelta(seconds=2),
                style="Default",
                name="host",
                text="Thanks for writing in.",
            ),
        ],
    )

    postprocess_subtitles(
        ass_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    with open(ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    events = [event for event in script.events if isinstance(event, pyass.Event)]
    assert events[0].text == '"This is a listener message."'
    assert events[1].text == "Thanks for writing in."


def test_postprocess_quotes_only_translated_line_in_bilingual_mode(tmp_path):
    ass_path = tmp_path / "translated.ass"
    _write_ass(
        ass_path,
        [
            pyass.Event(
                start=pyass.timedelta(seconds=0),
                end=pyass.timedelta(seconds=1),
                style="Default",
                name="listener_mail",
                text=r"{\fs24\a6}メールを送るのは初めてです。{\N}{\fs48\a2}This is my first message.",
            )
        ],
    )

    postprocess_subtitles(
        ass_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=True,
    )

    with open(ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    events = [event for event in script.events if isinstance(event, pyass.Event)]
    assert (
        events[0].text
        == r'{\fs24\a6}メールを送るのは初めてです。{\N}{\fs48\a2}"This is my first message."'
    )


def test_postprocess_collapses_double_outer_quotes_in_replace_mode(tmp_path):
    ass_path = tmp_path / "translated.ass"
    _write_ass(
        ass_path,
        [
            pyass.Event(
                start=pyass.timedelta(seconds=0),
                end=pyass.timedelta(seconds=1),
                style="Default",
                name="listener_mail",
                text='""This is a listener message.""',
            )
        ],
    )

    postprocess_subtitles(
        ass_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    with open(ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    events = [event for event in script.events if isinstance(event, pyass.Event)]
    assert events[0].text == '"This is a listener message."'


def test_postprocess_collapses_double_outer_quotes_on_bilingual_translation(tmp_path):
    ass_path = tmp_path / "translated.ass"
    _write_ass(
        ass_path,
        [
            pyass.Event(
                start=pyass.timedelta(seconds=0),
                end=pyass.timedelta(seconds=1),
                style="Default",
                name="listener_mail",
                text=r'{\fs24\a6}メールを送るのは初めてです。{\N}{\fs48\a2}""This is my first message.""',
            )
        ],
    )

    postprocess_subtitles(
        ass_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=True,
    )

    with open(ass_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    events = [event for event in script.events if isinstance(event, pyass.Event)]
    assert (
        events[0].text
        == r'{\fs24\a6}メールを送るのは初めてです。{\N}{\fs48\a2}"This is my first message."'
    )


def test_ensure_quoted_collapses_duplicate_quotes_on_first_and_last_visual_lines():
    text = r'""aaaaaa"\N"aaaaa"\N"aaaa""'

    assert _ensure_quoted(text) == r'"aaaaaa"\N"aaaaa"\N"aaaa"'


def test_ensure_quoted_collapses_duplicate_quotes_on_first_and_last_newline_lines():
    text = '""aaaaaa"\n"aaaaa"\n"aaaa""'

    assert _ensure_quoted(text) == '"aaaaaa"\n"aaaaa"\n"aaaa"'


def test_ensure_quoted_preserves_duplicate_quotes_at_interior_visual_line_edges():
    text = r'"aaaa"\N""bbbb""\N"cccc"'

    assert _ensure_quoted(text) == r'"aaaa"\N""bbbb""\N"cccc"'
