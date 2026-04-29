from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.pipeline.postprocess.main import _ensure_quoted, postprocess_subtitles


def _write_translated_document(path, cues: list[SubtitleCue]) -> None:
    document = SubtitleDocument(stage="translated", cues=cues)
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def test_postprocess_quotes_listener_mail_replace_mode(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールです。",
                translated_text="This is a listener message.",
                role="listener_mail",
            ),
            SubtitleCue(
                id="cue-00002",
                start_time=1,
                end_time=2,
                source_text="ありがとう。",
                translated_text="Thanks for writing in.",
                role="host",
            ),
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is a listener message."'
    assert document.cues[1].final_text == "Thanks for writing in."


def test_postprocess_quotes_only_translated_line_in_bilingual_mode(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールを送るのは初めてです。",
                translated_text="This is my first message.",
                role="listener_mail",
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=True,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is my first message."'


def test_postprocess_collapses_double_outer_quotes_in_replace_mode(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールです。",
                translated_text='""This is a listener message.""',
                role="listener_mail",
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=False,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is a listener message."'


def test_postprocess_collapses_double_outer_quotes_on_bilingual_translation(tmp_path):
    input_path = tmp_path / "translated.json"
    output_path = tmp_path / "postprocessed.json"
    _write_translated_document(
        input_path,
        [
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="メールを送るのは初めてです。",
                translated_text='""This is my first message.""',
                role="listener_mail",
            )
        ],
    )

    postprocess_subtitles(
        input_path,
        output_json_path=output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
        bilingual=True,
    )

    document = SubtitleDocument.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert document.cues[0].final_text == '"This is my first message."'


def test_ensure_quoted_collapses_duplicate_quotes_on_first_and_last_visual_lines():
    text = r'""aaaaaa"\N"aaaaa"\N"aaaa""'

    assert _ensure_quoted(text) == r'"aaaaaa"\N"aaaaa"\N"aaaa"'


def test_ensure_quoted_collapses_duplicate_quotes_on_first_and_last_newline_lines():
    text = '""aaaaaa"\n"aaaaa"\n"aaaa""'

    assert _ensure_quoted(text) == '"aaaaaa"\n"aaaaa"\n"aaaa"'


def test_ensure_quoted_preserves_duplicate_quotes_at_interior_visual_line_edges():
    text = r'"aaaa"\N""bbbb""\N"cccc"'

    assert _ensure_quoted(text) == r'"aaaa"\N""bbbb""\N"cccc"'
