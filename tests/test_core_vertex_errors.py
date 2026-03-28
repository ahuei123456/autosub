from types import SimpleNamespace

import pytest
from google.genai import types

from autosub.core.errors import (
    VertexBlockedResponseError,
    VertexRequestError,
    VertexResponseDiagnostics,
    VertexResponseParseError,
    VertexResponseShapeError,
)
from autosub.core.llm import BaseVertexLLM
from autosub.pipeline.translate.translator import VertexTranslator


class DummyVertexLLM(BaseVertexLLM):
    pass


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error

    def generate_content(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.models = FakeModels(response=response, error=error)


def _make_llm() -> DummyVertexLLM:
    return DummyVertexLLM(
        project_id="test-project",
        model="gemini-test",
        location="us-central1",
    )


def test_generate_structured_json_wraps_request_error(monkeypatch):
    llm = _make_llm()
    monkeypatch.setattr(
        llm,
        "_get_client",
        lambda: FakeClient(error=RuntimeError("simulated transport failure")),
    )

    with pytest.raises(VertexRequestError) as exc_info:
        llm._generate_structured_json(
            contents="[]",
            system_instruction="test",
            response_schema=list[dict],
            operation_name="Vertex test operation",
        )

    message = str(exc_info.value)
    assert "Vertex test operation request to Vertex failed" in message
    assert "project_id=test-project" in message
    assert "model=gemini-test" in message
    assert "location=us-central1" in message


def test_generate_structured_json_raises_blocked_response_error(monkeypatch):
    llm = _make_llm()
    response = SimpleNamespace(
        text=None,
        candidates=[
            types.Candidate(
                index=0,
                finish_reason=types.FinishReason.BLOCKLIST,
                finish_message="Forbidden term encountered.",
                token_count=12,
                safety_ratings=[
                    types.SafetyRating(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        probability=types.HarmProbability.HIGH,
                        blocked=True,
                    )
                ],
            )
        ],
        prompt_feedback=None,
        response_id="resp-1",
        model_version="gemini-test-version",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=5,
            candidates_token_count=12,
            total_token_count=17,
        ),
        sdk_http_response=types.HttpResponse(body='{"error":"blocked"}'),
    )
    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient(response=response))

    with pytest.raises(VertexBlockedResponseError) as exc_info:
        llm._generate_structured_json(
            contents="[]",
            system_instruction="test",
            response_schema=list[dict],
            operation_name="Vertex test operation",
        )

    message = str(exc_info.value)
    assert "Vertex test operation returned no text response." in message
    assert "finish_reasons=BLOCKLIST" in message
    assert "finish_messages=Forbidden term encountered." in message
    assert "response_id=resp-1" in message
    assert "tokens=prompt=5,candidates=12,total=17" in message


def test_generate_structured_json_raises_parse_error_with_preview(monkeypatch):
    llm = _make_llm()
    response = SimpleNamespace(
        text="not json",
        candidates=[
            types.Candidate(
                index=0,
                finish_reason=types.FinishReason.STOP,
                finish_message="Done.",
                token_count=7,
            )
        ],
        prompt_feedback=None,
        response_id="resp-2",
        model_version="gemini-test-version",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3,
            candidates_token_count=7,
            total_token_count=10,
        ),
        sdk_http_response=types.HttpResponse(body='{"error":"bad json"}'),
    )
    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient(response=response))

    with pytest.raises(VertexResponseParseError) as exc_info:
        llm._generate_structured_json(
            contents="[]",
            system_instruction="test",
            response_schema=list[dict],
            operation_name="Vertex test operation",
        )

    message = str(exc_info.value)
    assert "Vertex test operation returned invalid JSON" in message
    assert "response_id=resp-2" in message
    assert "text_preview=not json" in message
    assert "finish_reasons=STOP" in message


def test_vertex_translator_wraps_unexpected_json_shape(monkeypatch):
    translator = VertexTranslator(project_id="test-project")
    diagnostics = VertexResponseDiagnostics(response_id="resp-shape")

    monkeypatch.setattr(
        translator,
        "_generate_structured_json",
        lambda **kwargs: ([{"translated": "hello"}], diagnostics),
    )

    with pytest.raises(VertexResponseShapeError) as exc_info:
        translator.translate(["こんにちは"])

    message = str(exc_info.value)
    assert "unexpected structure" in message
    assert "response_id=resp-shape" in message
