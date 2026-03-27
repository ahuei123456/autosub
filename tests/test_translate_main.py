import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autosub.pipeline.translate.main import (
    _translate_with_retry,
    _translate_chunked,
    _load_checkpoint,
    _save_checkpoint,
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


# --- _translate_with_retry tests ---


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_retry_succeeds_on_first_attempt():
    translator = FakeTranslator()
    result = _translate_with_retry(translator, ["hello", "world"])
    assert result == ["translated:hello", "translated:world"]


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_retry_succeeds_after_failures():
    translator = FailNTimesTranslator(fail_count=2)
    result = _translate_with_retry(translator, ["hello"])
    assert result == ["translated:hello"]
    assert translator.attempts == 3


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_retry_raises_after_max_retries():
    translator = FailNTimesTranslator(fail_count=5)
    with pytest.raises(ConnectionError):
        _translate_with_retry(translator, ["hello"])
    assert translator.attempts == 3  # MAX_RETRIES = 3


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_retry_with_label_logs_correctly(caplog):
    translator = FailNTimesTranslator(fail_count=1)
    with caplog.at_level("WARNING"):
        _translate_with_retry(translator, ["hello"], label="Chunk 1")
    assert "Chunk 1" in caplog.text
    assert "Attempt 1 failed" in caplog.text


# --- _translate_chunked tests ---


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_chunked_splits_and_merges(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(5)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result = _translate_chunked(translator, texts, chunk_size=2, checkpoint_path=checkpoint_path)

    assert result == [f"translated:line{i}" for i in range(5)]
    # Checkpoint should still exist (caller is responsible for cleanup)
    assert checkpoint_path.exists()


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_chunked_retries_failing_chunk(tmp_path):
    translator = FailNTimesTranslator(fail_count=1)
    texts = ["a", "b", "c"]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result = _translate_chunked(translator, texts, chunk_size=2, checkpoint_path=checkpoint_path)

    assert result == ["translated:a", "translated:b", "translated:c"]
    assert translator.attempts > 1


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_chunked_preserves_order(tmp_path):
    translator = FakeTranslator()
    texts = [f"line{i}" for i in range(10)]
    checkpoint_path = tmp_path / "test.checkpoint.json"

    result = _translate_chunked(translator, texts, chunk_size=3, checkpoint_path=checkpoint_path)

    assert len(result) == 10
    for i in range(10):
        assert result[i] == f"translated:line{i}"


# --- Checkpoint tests ---


def test_save_and_load_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "test.checkpoint.json"
    data = {0: ["a", "b"], 1: ["c", "d"]}

    _save_checkpoint(checkpoint_path, data)
    loaded = _load_checkpoint(checkpoint_path)

    assert loaded == data


def test_load_checkpoint_missing_file(tmp_path):
    checkpoint_path = tmp_path / "nonexistent.json"
    assert _load_checkpoint(checkpoint_path) == {}


def test_load_checkpoint_corrupt_file(tmp_path):
    checkpoint_path = tmp_path / "corrupt.json"
    checkpoint_path.write_text("not valid json{{{")
    assert _load_checkpoint(checkpoint_path) == {}


def test_load_checkpoint_not_a_dict(tmp_path):
    checkpoint_path = tmp_path / "bad.json"
    checkpoint_path.write_text('["a", "b"]')
    assert _load_checkpoint(checkpoint_path) == {}


def test_load_checkpoint_skips_non_integer_keys(tmp_path):
    checkpoint_path = tmp_path / "bad_keys.json"
    checkpoint_path.write_text('{"0": ["a"], "foo": ["b"], "1": ["c"]}')
    result = _load_checkpoint(checkpoint_path)
    assert result == {0: ["a"], 1: ["c"]}


def test_load_checkpoint_skips_negative_keys(tmp_path):
    checkpoint_path = tmp_path / "neg.json"
    checkpoint_path.write_text('{"-1": ["a"], "0": ["b"]}')
    result = _load_checkpoint(checkpoint_path)
    assert result == {0: ["b"]}


def test_load_checkpoint_skips_empty_lists(tmp_path):
    checkpoint_path = tmp_path / "empty.json"
    checkpoint_path.write_text('{"0": ["a"], "1": []}')
    result = _load_checkpoint(checkpoint_path)
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_list_values(tmp_path):
    checkpoint_path = tmp_path / "bad_vals.json"
    checkpoint_path.write_text('{"0": ["a"], "1": "not a list", "2": 42}')
    result = _load_checkpoint(checkpoint_path)
    assert result == {0: ["a"]}


def test_load_checkpoint_skips_non_string_elements(tmp_path):
    checkpoint_path = tmp_path / "bad_elems.json"
    checkpoint_path.write_text('{"0": ["a", "b"], "1": [1, 2]}')
    result = _load_checkpoint(checkpoint_path)
    assert result == {0: ["a", "b"]}


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_chunked_resumes_from_checkpoint(tmp_path):
    """Simulate a previous run that completed chunks 0 and 1, then resume."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d", "e", "f"]

    # Pre-populate checkpoint with chunks 0 and 1 already done
    existing = {0: ["translated:a", "translated:b"], 1: ["translated:c", "translated:d"]}
    _save_checkpoint(checkpoint_path, existing)

    # Track which texts the translator actually receives
    translated_inputs = []
    original_translate = FakeTranslator.translate

    def tracking_translate(self, texts):
        translated_inputs.extend(texts)
        return original_translate(self, texts)

    translator = FakeTranslator()
    translator.translate = lambda texts: tracking_translate(translator, texts)

    result = _translate_chunked(translator, texts, chunk_size=2, checkpoint_path=checkpoint_path)

    # Should only translate chunk 2 (lines e, f)
    assert translated_inputs == ["e", "f"]
    # Full result should include checkpointed + new
    assert result == [
        "translated:a", "translated:b",
        "translated:c", "translated:d",
        "translated:e", "translated:f",
    ]


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_checkpoint_saved_after_each_chunk(tmp_path):
    """Verify checkpoint file is written after each chunk completes."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]

    saved_states = []
    original_save = _save_checkpoint.__wrapped__ if hasattr(_save_checkpoint, '__wrapped__') else None

    def tracking_save(path, completed):
        # Snapshot the state at each save
        saved_states.append(dict(completed))
        _save_checkpoint.__wrapped__(path, completed) if original_save else json.dump(
            completed, open(path, "w", encoding="utf-8"), ensure_ascii=False
        )

    translator = FakeTranslator()

    with patch("autosub.pipeline.translate.main._save_checkpoint") as mock_save:
        mock_save.side_effect = lambda path, completed: _save_checkpoint(path, completed)
        _translate_chunked(translator, texts, chunk_size=2, checkpoint_path=checkpoint_path)
        # Should be called once per chunk
        assert mock_save.call_count == 2


@patch("autosub.pipeline.translate.main.RETRY_BASE_DELAY", 0)
def test_chunked_all_checkpointed_skips_translation(tmp_path):
    """If all chunks are in the checkpoint, no translation calls should be made."""
    checkpoint_path = tmp_path / "test.checkpoint.json"
    texts = ["a", "b", "c", "d"]

    existing = {0: ["translated:a", "translated:b"], 1: ["translated:c", "translated:d"]}
    _save_checkpoint(checkpoint_path, existing)

    translator = MagicMock()

    result = _translate_chunked(translator, texts, chunk_size=2, checkpoint_path=checkpoint_path)

    translator.translate.assert_not_called()
    assert result == ["translated:a", "translated:b", "translated:c", "translated:d"]
