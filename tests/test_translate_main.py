from unittest.mock import MagicMock, patch

import pytest
import autosub.pipeline.translate.main as translate_main_module
import autosub.pipeline.translate.translator as translator_module

from autosub.core.schemas import SubtitleCue, SubtitleDocument
from autosub.pipeline.translate.main import (
    _compute_cue_fingerprint,
    _extract_corner_boundaries_from_cues,
    _translate_chunked,
    _load_checkpoint,
    _save_checkpoint,
    _write_error_report,
    translate_subtitles,
)


def _fake_translate_cues(self, cues: list[SubtitleCue]) -> list[str]:
    texts = [cue.normalized_source_text or cue.source_text for cue in cues]
    return self.translate(texts)


class FakeTranslator:
    """Translator that returns prefixed text."""

    def translate(self, texts: list[str]) -> list[str]:
        return [f"translated:{t}" for t in texts]

    translate_cues = _fake_translate_cues


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

    translate_cues = _fake_translate_cues


def _write_formatted_document(path):
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0.0,
                end_time=1.0,
                source_text="こんにちは",
                normalized_source_text="こんにちは",
            )
        ],
    )
    path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def _cues_from_texts(texts: list[str]) -> list[SubtitleCue]:
    return [
        SubtitleCue(
            id=f"cue-{index + 1:05d}",
            start_time=float(index),
            end_time=float(index + 1),
            source_text=text,
        )
        for index, text in enumerate(texts)
    ]


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

    result, splits = _translate_chunked(
        translator,
        _cues_from_texts(texts),
        chunk_size=2,
        checkpoint_path=checkpoint_path,
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
            translator,
            _cues_from_texts(texts),
            chunk_size=2,
            checkpoint_path=checkpoint_path,
        )

    assert translator.attempts == 1
    assert not checkpoint_path.exists()


def test_chunked_preserves_order(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(10)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result, splits = _translate_chunked(
        translator,
        _cues_from_texts(texts),
        chunk_size=3,
        checkpoint_path=checkpoint_path,
    )

    assert len(result) == 10
    for i in range(10):
        assert result[i] == f"translated:line{i}"


def test_chunked_uses_translate_cues_with_structured_metadata(tmp_path):
    class CueTrackingTranslator:
        def __init__(self):
            self.seen_roles: list[str | None] = []

        def translate_cues(self, cues: list[SubtitleCue]) -> list[str]:
            self.seen_roles.extend(cue.role for cue in cues)
            return [
                f"{cue.role}:{cue.normalized_source_text or cue.source_text}"
                for cue in cues
            ]

    translator = CueTrackingTranslator()
    cues = [
        SubtitleCue(
            id="cue-00001",
            start_time=0,
            end_time=1,
            source_text="a",
            role="host",
        ),
        SubtitleCue(
            id="cue-00002",
            start_time=1,
            end_time=2,
            source_text="b",
            role="listener_mail",
        ),
        SubtitleCue(
            id="cue-00003",
            start_time=2,
            end_time=3,
            source_text="c",
            role="host",
        ),
    ]

    result, _ = _translate_chunked(
        translator, cues, chunk_size=2, checkpoint_path=tmp_path / "checkpoint.json"
    )

    assert translator.seen_roles == ["host", "listener_mail", "host"]
    assert result == ["host:a", "listener_mail:b", "host:c"]


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

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a"], "foo": ["b"], "1": ["c"]}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"], 1: ["c"]}


def test_load_checkpoint_skips_negative_keys(tmp_path):
    checkpoint_path = tmp_path / "neg.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"-1": ["a"], "0": ["b"]}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["b"]}


def test_load_checkpoint_skips_empty_lists(tmp_path):
    checkpoint_path = tmp_path / "empty.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a"], "1": []}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_list_values(tmp_path):
    checkpoint_path = tmp_path / "bad_vals.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a"], "1": "not a list", "2": 42}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_string_elements(tmp_path):
    checkpoint_path = tmp_path / "bad_elems.json"
    import json

    json.dump(
        {"_fingerprint": "fp", "chunks": {"0": ["a", "b"], "1": [1, 2]}},
        open(checkpoint_path, "w"),
    )
    result = _load_checkpoint(checkpoint_path, "fp")
    assert result == {0: ["a", "b"]}


def test_chunked_resumes_from_checkpoint(tmp_path):
    """Simulate a previous run that completed chunks 0 and 1, then resume."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d", "e", "f"]
    fp = _compute_cue_fingerprint(
        _cues_from_texts(texts), chunk_size=2, corner_boundaries=None
    )

    # Pre-populate checkpoint with chunks 0 and 1 already done
    existing = {
        0: ["translated:a", "translated:b"],
        1: ["translated:c", "translated:d"],
    }
    _save_checkpoint(checkpoint_path, existing, fp)

    # Track which cues the translator actually receives
    translated_inputs = []
    original_translate_cues = FakeTranslator.translate_cues

    def tracking_translate_cues(self, cues):
        translated_inputs.extend(cue.source_text for cue in cues)
        return original_translate_cues(self, cues)

    translator = FakeTranslator()
    translator.translate_cues = lambda cues: tracking_translate_cues(translator, cues)

    result, splits = _translate_chunked(
        translator,
        _cues_from_texts(texts),
        chunk_size=2,
        checkpoint_path=checkpoint_path,
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
            translator,
            _cues_from_texts(texts),
            chunk_size=2,
            checkpoint_path=checkpoint_path,
        )
        # Should be called once per chunk
        assert mock_save.call_count == 2


def test_translate_subtitles_sets_llm_trace_path(tmp_path, monkeypatch):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

        translate_cues = _fake_translate_cues

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(input_json_path, output_json_path, engine="vertex")

    assert captured["trace_path"] == output_json_path.with_suffix(".llm_trace.jsonl")


def test_translate_subtitles_populates_translated_text_with_translate_cues(
    tmp_path, monkeypatch
):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            pass

        def translate(self, texts: list[str]) -> list[str]:
            raise AssertionError("translate_cues should be used")

        def translate_cues(self, cues: list[SubtitleCue]) -> list[str]:
            return [f"translated:{cue.source_text}" for cue in cues]

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(input_json_path, output_json_path, engine="vertex")

    document = SubtitleDocument.model_validate_json(
        output_json_path.read_text(encoding="utf-8")
    )
    assert document.stage == "translated"
    assert document.cues[0].translated_text == "translated:こんにちは"


def test_extract_corner_boundaries_from_cues_skips_empty_source_text():
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="",
                corner="ignored",
            ),
            SubtitleCue(
                id="cue-00002",
                start_time=1,
                end_time=2,
                source_text="intro",
            ),
            SubtitleCue(
                id="cue-00003",
                start_time=2,
                end_time=3,
                source_text="corner starts",
                corner="mail",
            ),
        ],
    )

    assert _extract_corner_boundaries_from_cues(document) == [1]


def test_extract_corner_boundaries_warns_for_empty_corner_cue(caplog):
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="",
                corner="ignored",
            )
        ],
    )

    with caplog.at_level("WARNING"):
        assert _extract_corner_boundaries_from_cues(document) == []

    assert "Ignoring corner boundary on empty cue cue-00001" in caplog.text


def test_subtitle_document_json_round_trip():
    document = SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id="cue-00001",
                start_time=0,
                end_time=1,
                source_text="こんにちは",
                normalized_source_text="こんにちは。",
                role="listener_mail",
            )
        ],
    )

    reloaded = SubtitleDocument.model_validate_json(document.model_dump_json())

    assert reloaded == document


def test_translate_subtitles_allows_anthropic_without_google_project(
    tmp_path, monkeypatch
):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

        translate_cues = _fake_translate_cues

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_json_path,
        output_json_path,
        engine="vertex",
        provider="anthropic",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "anthropic"
    assert captured["model"] is None


def test_translate_subtitles_allows_anthropic_vertex_with_google_project(
    tmp_path, monkeypatch
):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

        translate_cues = _fake_translate_cues

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_json_path,
        output_json_path,
        engine="vertex",
        provider="anthropic-vertex",
    )

    assert captured["project_id"] == "test-project"
    assert captured["provider"] == "anthropic-vertex"
    assert captured["model"] is None


def test_translate_subtitles_allows_openai_without_google_project(
    tmp_path, monkeypatch
):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

        translate_cues = _fake_translate_cues

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_json_path,
        output_json_path,
        engine="vertex",
        provider="openai",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "openai"
    assert captured["model"] is None


def test_translate_subtitles_allows_openrouter_without_google_project(
    tmp_path, monkeypatch
):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    captured: dict[str, object] = {}

    class FakeVertexTranslator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def translate(self, texts: list[str]) -> list[str]:
            return [f"translated:{text}" for text in texts]

        translate_cues = _fake_translate_cues

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)
    monkeypatch.setattr(translator_module, "VertexTranslator", FakeVertexTranslator)

    translate_subtitles(
        input_json_path,
        output_json_path,
        engine="vertex",
        provider="openrouter",
    )

    assert captured["project_id"] is None
    assert captured["provider"] == "openrouter"
    assert captured["model"] is None


def test_translate_subtitles_requires_google_project_for_anthropic_vertex(
    tmp_path, monkeypatch
):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", None)

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT is not set"):
        translate_subtitles(
            input_json_path,
            output_json_path,
            engine="vertex",
            provider="anthropic-vertex",
        )


def test_translate_subtitles_writes_error_file_on_failure(tmp_path, monkeypatch):
    input_json_path = tmp_path / "formatted.json"
    output_json_path = tmp_path / "translated.json"
    _write_formatted_document(input_json_path)

    class FailingVertexTranslator:
        def __init__(self, **kwargs):
            pass

        def translate(self, texts: list[str]) -> list[str]:
            raise RuntimeError("translation exploded")

        translate_cues = _fake_translate_cues

    monkeypatch.setattr(translate_main_module, "PROJECT_ID", "test-project")
    monkeypatch.setattr(translator_module, "VertexTranslator", FailingVertexTranslator)

    with pytest.raises(RuntimeError, match="translation exploded"):
        translate_subtitles(input_json_path, output_json_path, engine="vertex")

    error_path = output_json_path.with_suffix(".error.txt")
    report = error_path.read_text(encoding="utf-8")
    assert "Traceback" in report
    assert "RuntimeError: translation exploded" in report


def test_chunked_all_checkpointed_skips_translation(tmp_path):
    """If all chunks are in the checkpoint, no translation calls should be made."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]
    fp = _compute_cue_fingerprint(
        _cues_from_texts(texts), chunk_size=2, corner_boundaries=None
    )

    existing = {
        0: ["translated:a", "translated:b"],
        1: ["translated:c", "translated:d"],
    }
    _save_checkpoint(checkpoint_path, existing, fp)

    translator = MagicMock()
    translator.translate_cues = MagicMock()

    result, splits = _translate_chunked(
        translator,
        _cues_from_texts(texts),
        chunk_size=2,
        checkpoint_path=checkpoint_path,
    )

    translator.translate.assert_not_called()
    translator.translate_cues.assert_not_called()
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


def test_cue_fingerprint_changes_with_texts():
    fp1 = _compute_cue_fingerprint(
        _cues_from_texts(["a", "b", "c"]), chunk_size=2, corner_boundaries=None
    )
    fp2 = _compute_cue_fingerprint(
        _cues_from_texts(["b", "c"]), chunk_size=2, corner_boundaries=None
    )
    assert fp1 != fp2


def test_cue_fingerprint_changes_with_chunk_size():
    texts = ["a", "b", "c", "d"]
    fp1 = _compute_cue_fingerprint(
        _cues_from_texts(texts), chunk_size=2, corner_boundaries=None
    )
    fp2 = _compute_cue_fingerprint(
        _cues_from_texts(texts), chunk_size=3, corner_boundaries=None
    )
    assert fp1 != fp2


def test_cue_fingerprint_changes_with_translation_metadata():
    base = SubtitleCue(
        id="cue-00001",
        start_time=0,
        end_time=1,
        source_text="same text",
        role="host",
    )
    changed_role = base.model_copy(update={"role": "listener_mail"})

    fp1 = _compute_cue_fingerprint([base], chunk_size=2, corner_boundaries=None)
    fp2 = _compute_cue_fingerprint([changed_role], chunk_size=2, corner_boundaries=None)

    assert fp1 != fp2
