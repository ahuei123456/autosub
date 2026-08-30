"""Vocabulary hints cannot reach a subtitle run, on either Chirp model.

SpeechAdaptation and enable_word_time_offsets are mutually exclusive in
Speech-to-Text v2. Subtitle timing needs the offsets, so the hints always lose:
Chirp 3 returns 404 when both are set, and Chirp 2 accepts the request but
silently ignores the phrases. Either way the caller should be told.
"""

import logging
from types import SimpleNamespace

import pytest

from autosub.pipeline.transcribe import api


class _FakeSpeechClient:
    """Captures the request instead of calling the API."""

    last_config = None

    def __init__(self, *args, **kwargs):
        pass

    def recognize(self, request):
        type(self).last_config = request.config
        return SimpleNamespace(results=[])

    def batch_recognize(self, request):
        type(self).last_config = request.config
        return SimpleNamespace(results={})


@pytest.fixture
def fake_client(monkeypatch):
    _FakeSpeechClient.last_config = None
    monkeypatch.setattr(api.speech_v2, "SpeechClient", _FakeSpeechClient)
    return _FakeSpeechClient


@pytest.mark.parametrize("model", ["chirp_2", "chirp_3"])
def test_vocabulary_hints_warn_on_both_models(fake_client, caplog, model):
    """The warning used to fire only for Chirp 3, implying Chirp 2 was fine."""
    with caplog.at_level(logging.WARNING, logger=api.logger.name):
        api.transcribe_local_file(
            audio_content=b"",
            project_id="test-project",
            vocabulary=["ニーゴ", "セカイ"],
            model=model,
        )

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("will NOT affect this transcript" in w for w in warnings), warnings
    # The count is interpolated so the user can tell which hints were dropped.
    assert any("2 vocabulary hints" in w for w in warnings), warnings


def test_no_warning_without_vocabulary(fake_client, caplog):
    with caplog.at_level(logging.WARNING, logger=api.logger.name):
        api.transcribe_local_file(
            audio_content=b"",
            project_id="test-project",
            vocabulary=None,
            model="chirp_2",
        )

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("vocabulary hints" in w for w in warnings), warnings


def test_adaptation_is_never_sent_on_chirp_3(fake_client):
    """Chirp 3 returns 404 when adaptation and word timings are both set."""
    api.transcribe_local_file(
        audio_content=b"",
        project_id="test-project",
        vocabulary=["ニーゴ"],
        model="chirp_3",
    )

    assert not fake_client.last_config.adaptation.phrase_sets


def test_word_time_offsets_always_win(fake_client):
    """Timing drives every subtitle line, so it must survive the hints."""
    api.transcribe_local_file(
        audio_content=b"",
        project_id="test-project",
        vocabulary=["ニーゴ"],
        model="chirp_2",
    )

    assert fake_client.last_config.features.enable_word_time_offsets
