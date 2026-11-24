"""Tests for the gemini-3 thinking-explosion mitigation.

Covers the per-chunk reasoning-effort fallback: when a chunk fails at high/medium
reasoning (thinking saturates the output budget), retry just that chunk at LOW.
"""

import pytest

from autosub.core.llm import ReasoningEffort
from autosub.core.schemas import SubtitleCue
from autosub.pipeline.translate import main as translate_main


# --- per-chunk reasoning fallback ------------------------------------------


def _cues(*texts: str) -> list[SubtitleCue]:
    return [
        SubtitleCue(
            id=f"cue-{i + 1:05d}",
            start_time=float(i),
            end_time=float(i + 1),
            source_text=text,
        )
        for i, text in enumerate(texts)
    ]


class _FakeTranslator:
    """Records translate_cues() calls and the reasoning_effort active for each."""

    def __init__(self, reasoning_effort, fail_first=False):
        self.reasoning_effort = reasoning_effort
        self.fail_first = fail_first
        self.calls = []  # reasoning_effort active at each call

    def translate_cues(self, cues: list[SubtitleCue]) -> list[str]:
        self.calls.append(self.reasoning_effort)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("Model token limit (65536) exceeded")
        return [f"t:{c.source_text}" for c in cues]


def test_fallback_retries_at_low_and_restores_effort():
    t = _FakeTranslator(ReasoningEffort.MEDIUM, fail_first=True)
    out = translate_main._translate_chunk_with_fallback(t, _cues("a", "b"), 0, 1)
    assert out == ["t:a", "t:b"]
    # first attempt at MEDIUM, retry at LOW
    assert t.calls == [ReasoningEffort.MEDIUM, ReasoningEffort.LOW]
    # original effort restored afterwards
    assert t.reasoning_effort == ReasoningEffort.MEDIUM


def test_no_fallback_when_already_low_reraises():
    t = _FakeTranslator(ReasoningEffort.LOW, fail_first=True)
    with pytest.raises(RuntimeError):
        translate_main._translate_chunk_with_fallback(t, _cues("a"), 0, 1)
    # no retry attempted
    assert t.calls == [ReasoningEffort.LOW]


def test_success_path_no_retry():
    t = _FakeTranslator(ReasoningEffort.HIGH, fail_first=False)
    out = translate_main._translate_chunk_with_fallback(t, _cues("a"), 0, 1)
    assert out == ["t:a"]
    assert t.calls == [ReasoningEffort.HIGH]
    assert t.reasoning_effort == ReasoningEffort.HIGH
