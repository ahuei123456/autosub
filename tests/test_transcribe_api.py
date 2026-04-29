from types import SimpleNamespace
import logging

from autosub.pipeline.transcribe import api as transcribe_api


def test_transcribe_uri_polls_and_logs_heartbeat_until_complete(monkeypatch, caplog):
    gcs_uri = "gs://bucket/audio.wav"
    clock = {"now": 0.0}

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(seconds):
        clock["now"] += seconds

    class FakeOperation:
        def __init__(self):
            self.operation = SimpleNamespace(name="operations/123")

        def done(self):
            return clock["now"] >= 12.0

        def result(self):
            return "response"

    class FakeSpeechClient:
        def __init__(self, *args, **kwargs):
            pass

        def batch_recognize(self, request):
            return FakeOperation()

    monkeypatch.setattr(transcribe_api.speech_v2, "SpeechClient", FakeSpeechClient)
    monkeypatch.setattr(transcribe_api.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(transcribe_api.time, "sleep", fake_sleep)
    monkeypatch.setattr(transcribe_api, "DEFAULT_BATCH_POLL_INTERVAL_SECONDS", 5.0)
    monkeypatch.setattr(transcribe_api, "DEFAULT_BATCH_HEARTBEAT_SECONDS", 10.0)

    with caplog.at_level(logging.INFO):
        response = transcribe_api.transcribe_uri(gcs_uri, "project-id")

    assert response == "response"
    assert any(
        "Submitted Chirp 2 batch job operations/123" in m for m in caplog.messages
    )
    assert any(
        "Still waiting on Chirp 2 batch job operations/123" in m
        for m in caplog.messages
    )
    assert any("completed in 0m 15s" in m for m in caplog.messages)


def test_transcribe_uri_wraps_operation_failures_with_context(monkeypatch):
    gcs_uri = "gs://bucket/audio.wav"

    class FakeOperation:
        def __init__(self):
            self.operation = SimpleNamespace(name="operations/boom")

        def done(self):
            return True

        def result(self):
            raise RuntimeError("backend boom")

    class FakeSpeechClient:
        def __init__(self, *args, **kwargs):
            pass

        def batch_recognize(self, request):
            return FakeOperation()

    monkeypatch.setattr(transcribe_api.speech_v2, "SpeechClient", FakeSpeechClient)

    try:
        transcribe_api.transcribe_uri(gcs_uri, "project-id")
        assert False, "Expected chirp_2 batch failures to be wrapped."
    except RuntimeError as exc:
        assert gcs_uri in str(exc)
        assert "backend boom" in str(exc)
