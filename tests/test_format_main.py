import json

import pyass
import autosub.extensions.radio_discourse.main as radio_discourse_main

from autosub.pipeline.format.main import format_subtitles


REAL_WHISPERX_INTRO_WORDS = [
    {"word": "こ", "start_time": 15.343, "end_time": 15.443, "confidence": 0.796},
    {"word": "ん", "start_time": 15.443, "end_time": 15.543, "confidence": 0.777},
    {"word": "ば", "start_time": 15.543, "end_time": 15.643, "confidence": 0.8},
    {"word": "ん", "start_time": 15.643, "end_time": 15.763, "confidence": 0.822},
    {"word": "は", "start_time": 15.763, "end_time": 16.023, "confidence": 0.88},
    {"word": "、", "start_time": 16.023, "end_time": 16.163, "confidence": 0.888},
    {"word": "声", "start_time": 16.163, "end_time": 16.383, "confidence": 0.907},
    {"word": "優", "start_time": 16.383, "end_time": 16.503, "confidence": 0.833},
    {"word": "の", "start_time": 16.503, "end_time": 16.883, "confidence": 0.941},
    {"word": "鈴", "start_time": 16.883, "end_time": 17.003, "confidence": 0.827},
    {"word": "原", "start_time": 17.003, "end_time": 17.083, "confidence": 0.725},
    {"word": "の", "start_time": 17.083, "end_time": 17.323, "confidence": 0.893},
    {"word": "ぞ", "start_time": 17.323, "end_time": 17.483, "confidence": 0.869},
    {"word": "み", "start_time": 17.483, "end_time": 17.604, "confidence": 0.989},
    {"word": "で", "start_time": 17.604, "end_time": 17.824, "confidence": 0.975},
    {"word": "す", "start_time": 17.824, "end_time": 18.224, "confidence": 0.996},
]


def test_format_subtitles_does_not_insert_ass_line_breaks(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [
        {
            "word": "語",
            "start_time": index * 0.1,
            "end_time": index * 0.1 + 0.05,
        }
        for index in range(20)
    ]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    format_subtitles(transcript_path, output_path)

    ass_text = output_path.read_text(encoding="utf-8")
    assert r"\N" not in ass_text


def test_format_subtitles_applies_radio_discourse_extension_and_preserves_role(
    tmp_path,
):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [
        {"word": "おすすめ", "start_time": 0.0, "end_time": 0.6},
        {"word": "です", "start_time": 0.6, "end_time": 1.0},
        {"word": "といただきました。", "start_time": 1.0, "end_time": 2.0},
    ]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    format_subtitles(
        transcript_path,
        output_path,
        extensions_config={"radio_discourse": {"enabled": True}},
    )

    with open(output_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    dialogue_events = [
        event for event in script.events if isinstance(event, pyass.Event)
    ]
    assert len(dialogue_events) == 2
    assert dialogue_events[0].text == "おすすめです。"
    assert dialogue_events[0].name == "listener_mail"
    assert dialogue_events[1].text == "といただきました。"
    assert dialogue_events[1].name == "host_meta"


def test_format_subtitles_sets_radio_discourse_trace_path(tmp_path, monkeypatch):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [{"word": "こんにちは", "start_time": 0.0, "end_time": 1.0}]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_apply(lines, config):
        captured["llm_trace_path"] = config.get("llm_trace_path")
        return lines

    monkeypatch.setattr(radio_discourse_main, "apply_radio_discourse", fake_apply)

    format_subtitles(
        transcript_path,
        output_path,
        extensions_config={"radio_discourse": {"enabled": True, "engine": "hybrid"}},
    )

    assert captured["llm_trace_path"] == output_path.with_suffix(
        ".radio_discourse.llm_trace.jsonl"
    )


def test_format_subtitles_sets_llm_normalizer_trace_path(tmp_path, monkeypatch):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    words = [{"word": "こんにちは", "start_time": 0.0, "end_time": 1.0}]
    transcript_path.write_text(
        json.dumps({"words": words}, ensure_ascii=False),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_apply(lines, config):
        captured["engine"] = config.get("engine")
        captured["llm_trace_path"] = config.get("llm_trace_path")
        captured["edit_audit_path"] = config.get("edit_audit_path")
        return lines

    monkeypatch.setattr("autosub.pipeline.format.main.apply_normalization", fake_apply)

    format_subtitles(
        transcript_path,
        output_path,
        normalizer_config={
            "engine": "llm",
            "terms": [{"value": "こんにちは", "explanation": "Greeting."}],
        },
    )

    assert captured["engine"] == "llm"
    assert captured["llm_trace_path"] == output_path.with_suffix(
        ".normalizer.llm_trace.jsonl"
    )
    assert captured["edit_audit_path"] == output_path.with_suffix(
        ".normalizer.edit_audit.tsv"
    )


def test_format_subtitles_prefers_whisperx_segments_when_present(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    transcript_path.write_text(
        json.dumps(
            {
                # Real WhisperX sample from nonshichotto/143/whisper.json.
                "words": REAL_WHISPERX_INTRO_WORDS,
                "segments": [
                    {
                        "text": "こんばんは、声優の鈴原のぞみです",
                        "start_time": 15.343,
                        "end_time": 18.224,
                        "words": REAL_WHISPERX_INTRO_WORDS,
                        "kind": "sentence",
                    }
                ],
                "metadata": {
                    "backend": "whisperx",
                    "language": "ja",
                    "model": "large-v2",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    format_subtitles(transcript_path, output_path)

    with open(output_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    dialogue_events = [
        event for event in script.events if isinstance(event, pyass.Event)
    ]
    assert len(dialogue_events) == 1
    assert dialogue_events[0].text == "こんばんは、声優の鈴原のぞみです"


def test_format_subtitles_keeps_legacy_word_chunking_for_non_whisperx(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    transcript_path.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "おすすめ", "start_time": 0.0, "end_time": 0.6},
                    {"word": "です。", "start_time": 0.6, "end_time": 1.0},
                ],
                "segments": [
                    {
                        "text": "SHOULD NOT USE",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "words": [],
                        "kind": "result",
                    }
                ],
                "metadata": {
                    "backend": "chirp_2",
                    "language": "ja-JP",
                    "model": "chirp_2",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    format_subtitles(transcript_path, output_path)

    with open(output_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    dialogue_events = [
        event for event in script.events if isinstance(event, pyass.Event)
    ]
    assert len(dialogue_events) == 1
    assert dialogue_events[0].text == "おすすめです。"


def test_format_subtitles_merges_multiple_inputs_after_initial_line_generation(
    tmp_path,
):
    transcript_path_a = tmp_path / "transcript_a.json"
    transcript_path_b = tmp_path / "transcript_b.json"
    output_path = tmp_path / "original.ass"

    transcript_path_a.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "IGNORED", "start_time": 10.0, "end_time": 11.0},
                ],
                "segments": [
                    {
                        "text": "whisperx segment",
                        "start_time": 10.0,
                        "end_time": 11.0,
                        "words": [
                            {"word": "whisperx", "start_time": 10.0, "end_time": 10.5},
                            {"word": "segment", "start_time": 10.5, "end_time": 11.0},
                        ],
                        "kind": "sentence",
                    }
                ],
                "metadata": {"backend": "whisperx", "language": "ja"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transcript_path_b.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "おすすめ", "start_time": 0.0, "end_time": 0.6},
                    {"word": "です。", "start_time": 0.6, "end_time": 1.0},
                ],
                "metadata": {"backend": "chirp_2", "language": "ja-JP"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    format_subtitles([transcript_path_a, transcript_path_b], output_path)

    with open(output_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    dialogue_events = [
        event for event in script.events if isinstance(event, pyass.Event)
    ]
    assert [event.text for event in dialogue_events] == [
        "おすすめです。",
        "whisperx segment",
    ]


def test_format_subtitles_warns_when_same_input_file_is_passed_twice(tmp_path, caplog):
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "original.ass"

    transcript_path.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "おすすめ", "start_time": 0.0, "end_time": 0.6},
                    {"word": "です。", "start_time": 0.6, "end_time": 1.0},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    format_subtitles([transcript_path, transcript_path], output_path)

    assert "Duplicate transcript input detected" in caplog.text

    with open(output_path, "r", encoding="utf-8") as handle:
        script = pyass.load(handle)

    dialogue_events = [
        event for event in script.events if isinstance(event, pyass.Event)
    ]
    assert len(dialogue_events) == 2


def test_format_subtitles_warns_when_input_time_ranges_overlap(tmp_path, caplog):
    transcript_path_a = tmp_path / "transcript_a.json"
    transcript_path_b = tmp_path / "transcript_b.json"
    output_path = tmp_path / "original.ass"

    transcript_path_a.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "first", "start_time": 0.0, "end_time": 1.0},
                    {"word": "range", "start_time": 1.0, "end_time": 2.0},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    transcript_path_b.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "overlap", "start_time": 1.5, "end_time": 2.0},
                    {"word": "range", "start_time": 2.0, "end_time": 2.5},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    format_subtitles([transcript_path_a, transcript_path_b], output_path)

    assert "Transcript time ranges overlap" in caplog.text
    assert "lines will be interleaved without dedup" in caplog.text


def test_format_subtitles_warns_when_input_produces_zero_lines(tmp_path, caplog):
    transcript_path = tmp_path / "empty_transcript.json"
    output_path = tmp_path / "original.ass"

    transcript_path.write_text(json.dumps({}), encoding="utf-8")

    format_subtitles(transcript_path, output_path)

    assert "Transcript produced zero initial subtitle lines" in caplog.text
