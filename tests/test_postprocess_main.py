import pytest

from autosub.core.schemas import SubtitleCue, SubtitleDocument, TranscribedWord
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


def test_postprocess_preserves_source_words(tmp_path):
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
                words=[
                    TranscribedWord(word="メール", start_time=0.0, end_time=0.5),
                    TranscribedWord(word="です。", start_time=0.5, end_time=1.0),
                ],
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
    assert document.stage == "postprocessed"
    assert document.cues[0].final_text == '"This is a listener message."'
    assert [word.word for word in document.cues[0].words] == ["メール", "です。"]


def test_postprocess_requires_translated_document(tmp_path):
    input_path = tmp_path / "formatted.json"
    output_path = tmp_path / "postprocessed.json"
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="source",
            )
        ],
    )
    input_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(
        ValueError, match="postprocess expects stage='translated', got 'formatted'"
    ):
        postprocess_subtitles(input_path, output_json_path=output_path)


def test_postprocess_preserves_and_renders_chunk_boundaries(tmp_path):
    input_path = tmp_path / "translated.json"
    output_json_path = tmp_path / "postprocessed.json"
    output_ass_path = tmp_path / "postprocessed.ass"
    document = SubtitleDocument(
        stage="translated",
        chunk_boundaries=[1],
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="first",
                translated_text="First.",
            ),
            SubtitleCue(
                id="cue-00002",
                start_time=1,
                end_time=2,
                source_text="second",
                translated_text="Second.",
            ),
        ],
    )
    input_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    postprocess_subtitles(
        input_path,
        output_json_path=output_json_path,
        output_ass_path=output_ass_path,
        bilingual=False,
    )

    processed = SubtitleDocument.model_validate_json(
        output_json_path.read_text(encoding="utf-8")
    )
    assert processed.chunk_boundaries == [1]
    assert (
        "[autosub] Chunk boundary - review translation around this line"
        in output_ass_path.read_text(encoding="utf-8")
    )


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
