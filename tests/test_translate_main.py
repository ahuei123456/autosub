from unittest.mock import MagicMock, patch

import pytest
import autosub.pipeline.translate.main as translate_main_module
import autosub.pipeline.translate.translator as translator_module

from autosub.pipeline.translate.main import (
    _compute_fingerprint,
    _translate_chunked,
    _load_checkpoint,
    _save_checkpoint,
    _write_error_report,
    translate_subtitles,
)


class FakeTranslator:
    """Translator that returns prefixed text."""

    def translate(self, texts: list[str]) -> list[str]:
        return [f"translated:{t}" for t in texts]


class FailNTimesTranslator:
    """Translator that fails N times then succeeds."""

    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.attempts = 0

    def translate(self, texts: list[str]) -> list[str]:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ConnectionError("Server disconnected without sending a response.")
        return [f"translated:{t}" for t in texts]


# --- Error report tests ---


def test_write_error_report_includes_traceback(tmp_path):
    error_path = tmp_path / "translated.error.txt"

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        _write_error_report(error_path, exc)

    report = error_path.read_text(encoding="utf-8")
    assert "Traceback" in report
    assert "RuntimeError: boom" in report


# --- _translate_chunked tests ---


def test_chunked_splits_and_merges(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(5)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result = _translate_chunked(
        translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
    )

    assert result == [f"translated:line{i}" for i in range(5)]
    # Checkpoint should still exist (caller is responsible for cleanup)
    assert checkpoint_path.exists()


def test_chunked_fails_fast_on_error(tmp_path):
    translator = FailNTimesTranslator(fail_count=1)
    texts = ["a", "b", "c"]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    with pytest.raises(ConnectionError):
        _translate_chunked(
            translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
        )

    assert translator.attempts == 1
    assert not checkpoint_path.exists()


def test_chunked_preserves_order(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(10)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result = _translate_chunked(
        translator, texts, chunk_size=3, checkpoint_path=checkpoint_path
    )

    assert len(result) == 10
    for i in range(10):
        assert result[i] == f"translated:line{i}"


# --- Checkpoint tests ---


def test_save_and_load_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "test.checkpoint.json"
    data = {0: ["a", "b"], 1: ["c", "d"]}
    fp = "test_fingerprint"

    _save_checkpoint(checkpoint_path, data, fp)
    loaded = _load_checkpoint(checkpoint_path, fp)

    assert loaded == data


def test_load_checkpoint_missing_file(tmp_path):
    checkpoint_path = tmp_path / "nonexistent.json"
    assert _load_checkpoint(checkpoint_path, "any") == {}


def test_load_checkpoint_corrupt_file(tmp_path):
    checkpoint_path = tmp_path / "corrupt.json"
    checkpoint_path.write_text("not valid json{{{")
    assert _load_checkpoint(checkpoint_path, "any") == {}


def test_load_checkpoint_not_a_dict(tmp_path):
    checkpoint_path = tmp_path / "bad.json"
    checkpoint_path.write_text('["a", "b"]')
    assert _load_checkpoint(checkpoint_path, "any") == {}


def test_load_checkpoint_skips_non_integer_keys(tmp_path):
    checkpoint_path = tmp_path / "bad_keys.json"
    import json
    json.dump({"_fingerprint": "fp", "chunks": {"0": ["a"], "foo": ["b"], "1": ["c"]}},
              open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"], 1: ["c"]}


def test_load_checkpoint_skips_negative_keys(tmp_path):
    checkpoint_path = tmp_path / "neg.json"
    import json
    json.dump({"_fingerprint": "fp", "chunks": {"-1": ["a"], "0": ["b"]}},
              open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["b"]}


def test_load_checkpoint_skips_empty_lists(tmp_path):
    checkpoint_path = tmp_path / "empty.json"
    import json
    json.dump({"_fingerprint": "fp", "chunks": {"0": ["a"], "1": []}},
              open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_list_values(tmp_path):
    checkpoint_path = tmp_path / "bad_vals.json"
    import json
    json.dump({"_fingerprint": "fp", "chunks": {"0": ["a"], "1": "not a list", "2": 42}},
              open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_string_elements(tmp_path):
    checkpoint_path = tmp_path / "bad_elems.json"
    import json
    json.dump({"_fingerprint": "fp", "chunks": {"0": ["a", "b"], "1": [1, 2]}},
              open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a", "b"]}


def test_chunked_resumes_from_checkpoint(tmp_path):
    """Simulate a previous run that completed chunks 0 and 1, then resume."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d", "e", "f"]
    fp = _compute_fingerprint(texts, chunk_size=2, corner_cues=None)

    # Pre-populate checkpoint with chunks 0 and 1 already done
    existing = {
        0: ["translated:a", "translated:b"],
        1: ["translated:c", "translated:d"],
    }
    _save_checkpoint(checkpoint_path, existing, fp)

    # Track which texts the translator actually receives
    translated_inputs = []
    original_translate = FakeTranslator.translate

    def tracking_translate(self, texts):
        translated_inputs.extend(texts)
        return original_translate(self, texts)

    translator = FakeTranslator()
    translator.translate = lambda texts: tracking_translate(translator, texts)

    result = _translate_chunked(
        translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
    )

    # Should only translate chunk 2 (lines e, f)
    assert translated_inputs == ["e", "f"]
    # Full result should include checkpointed + new
    assert result == [
        "translated:a",
        "translated:b",
        "translated:c",
        "translated:d",
        "translated:e",
        "translated:f",
    ]


def test_checkpoint_saved_after_each_chunk(tmp_path):
    """Verify checkpoint file is written after each chunk completes."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]

    translator = FakeTranslator()

    with patch("autosub.pipeline.translate.main._save_checkpoint") as mock_save:
        mock_save.side_effect = lambda path, completed, fp: _save_checkpoint(
            path, completed, fp
        )
        _translate_chunked(
            translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
        )
        # Should be called once per chunk
        assert mock_save.call_count == 2


def test_translate_subtitles_sets_llm_trace_path(tmp_path, monkeypatch):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(input_ass_path, output_ass_path, engine="vertex")

    assert captured["trace_path"] == output_ass_path.with_suffix(".llm_trace.jsonl")


def test_translate_subtitles_allows_anthropic_without_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="anthropic",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "anthropic"
    assert captured["model"] is None


def test_translate_subtitles_allows_openai_without_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="openai",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "openai"
    assert captured["model"] is None


def test_translate_subtitles_allows_openrouter_without_google_project(
    tmp_path, monkeypatch
):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_ass_path,
        output_ass_path,
        engine="vertex",
        provider="openrouter",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "openrouter"
    assert captured["model"] is None


def test_translate_subtitles_writes_error_file_on_failure(tmp_path, monkeypatch):
    input_ass_path = tmp_path / "original.ass"
    output_ass_path = tmp_path / "translated.ass"
    input_ass_path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )

    class FailingVertexTranslator:
        def __init__(self, **kwargs):
            pass

        def translate(self, texts: list[str]) -> list[str]:
            raise RuntimeError("translation exploded")

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FailingVertexTranslator)

    with pytest.raises(RuntimeError, match="translation exploded"):
        translate_subtitles(input_ass_path, output_ass_path, engine="vertex")

    error_path = output_ass_path.with_suffix(".error.txt")
    report = error_path.read_text(encoding="utf-8")
    assert "Traceback" in report
    assert "RuntimeError: translation exploded" in report


def test_chunked_all_checkpointed_skips_translation(tmp_path):
    """If all chunks are in the checkpoint, no translation calls should be made."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]
    fp = _compute_fingerprint(texts, chunk_size=2, corner_cues=None)

    existing = {
        0: ["translated:a", "translated:b"],
        1: ["translated:c", "translated:d"],
    }
    _save_checkpoint(checkpoint_path, existing, fp)

    translator = MagicMock()

    result = _translate_chunked(
        translator, texts, chunk_size=2, checkpoint_path=checkpoint_path
    )

    translator.translate.assert_not_called()
    assert result == ["translated:a", "translated:b", "translated:c", "translated:d"]


# --- Fingerprint tests ---


def test_load_checkpoint_fingerprint_mismatch(tmp_path):
    """Checkpoint with wrong fingerprint is discarded."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    _save_checkpoint(checkpoint_path, {0: ["a"]}, "fingerprint_aaa")
    result = _load_checkpoint(checkpoint_path, "fingerprint_bbb")
    assert result == {}


def test_load_checkpoint_legacy_format_discarded(tmp_path):
    """Old-format checkpoint (no _fingerprint) is discarded."""
    checkpoint_path = tmp_path / "legacy.json"
    import json
    json.dump({"0": ["a", "b"], "1": ["c"]}, open(checkpoint_path, "w"))
    result = _load_checkpoint(checkpoint_path, "any_fingerprint")
    assert result == {}


def test_fingerprint_changes_with_texts():
    fp1 = _compute_fingerprint(["a", "b", "c"], chunk_size=2, corner_cues=None)
    fp2 = _compute_fingerprint(["b", "c"], chunk_size=2, corner_cues=None)
    assert fp1 != fp2


def test_fingerprint_changes_with_chunk_size():
    texts = ["a", "b", "c", "d"]
    fp1 = _compute_fingerprint(texts, chunk_size=2, corner_cues=None)
    fp2 = _compute_fingerprint(texts, chunk_size=3, corner_cues=None)
    assert fp1 != fp2
