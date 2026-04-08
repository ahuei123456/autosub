from __future__ import annotations

from dataclasses import dataclass


def _truncate(value: str | None, limit: int = 240) -> str | None:
    if value is None:
        return None

    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


@dataclass(slots=True)
class VertexResponseDiagnostics:
    response_id: str | None = None
    model_version: str | None = None
    prompt_block_reason: str | None = None
    prompt_block_reason_message: str | None = None
    prompt_safety_ratings: tuple[str, ...] = ()
    candidate_finish_reasons: tuple[str, ...] = ()
    candidate_finish_messages: tuple[str, ...] = ()
    candidate_token_counts: tuple[int, ...] = ()
    candidate_safety_ratings: tuple[str, ...] = ()
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None
    total_token_count: int | None = None
    thoughts_token_count: int | None = None
    thinking_text: str | None = None
    text_preview: str | None = None
    http_body_preview: str | None = None

    def summary_parts(self) -> list[str]:
        parts: list[str] = []
        if self.response_id:
            parts.append(f"response_id={self.response_id}")
        if self.model_version:
            parts.append(f"model_version={self.model_version}")
        if self.prompt_block_reason:
            parts.append(f"prompt_block_reason={self.prompt_block_reason}")
        if self.prompt_block_reason_message:
            parts.append(
                f"prompt_block_message={_truncate(self.prompt_block_reason_message, 120)}"
            )
        if self.candidate_finish_reasons:
            parts.append(
                "finish_reasons="
                + ",".join(reason for reason in self.candidate_finish_reasons if reason)
            )
        if self.candidate_finish_messages:
            parts.append(
                "finish_messages="
                + " | ".join(
                    message
                    for message in (
                        _truncate(item, 80) for item in self.candidate_finish_messages
                    )
                    if message
                )
            )
        if self.prompt_safety_ratings:
            parts.append("prompt_safety=" + "; ".join(self.prompt_safety_ratings))
        if self.candidate_safety_ratings:
            parts.append("candidate_safety=" + "; ".join(self.candidate_safety_ratings))
        if self.candidate_token_counts:
            parts.append(
                "candidate_token_counts="
                + ",".join(str(count) for count in self.candidate_token_counts)
            )

        token_parts = []
        if self.prompt_token_count is not None:
            token_parts.append(f"prompt={self.prompt_token_count}")
        if self.candidates_token_count is not None:
            token_parts.append(f"candidates={self.candidates_token_count}")
        if self.thoughts_token_count is not None:
            token_parts.append(f"thoughts={self.thoughts_token_count}")
        if self.total_token_count is not None:
            token_parts.append(f"total={self.total_token_count}")
        if token_parts:
            parts.append("tokens=" + ",".join(token_parts))

        if self.text_preview:
            parts.append(f"text_preview={self.text_preview}")

        return parts


class AutosubError(Exception):
    """Base exception for autosub-specific failures."""


class VertexError(AutosubError):
    def __init__(
        self,
        message: str,
        *,
        project_id: str | None = None,
        model: str | None = None,
        location: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.project_id = project_id
        self.model = model
        self.location = location

    def _context_parts(self) -> list[str]:
        parts: list[str] = []
        if self.project_id:
            parts.append(f"project_id={self.project_id}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.location:
            parts.append(f"location={self.location}")
        return parts

    def __str__(self) -> str:
        parts = self._context_parts()
        if not parts:
            return self.message
        return f"{self.message} ({'; '.join(parts)})"


class VertexRequestError(VertexError):
    """Vertex request failed before a usable response body was produced."""


class VertexResponseError(VertexError):
    def __init__(
        self,
        message: str,
        *,
        diagnostics: VertexResponseDiagnostics,
        project_id: str | None = None,
        model: str | None = None,
        location: str | None = None,
    ):
        super().__init__(
            message,
            project_id=project_id,
            model=model,
            location=location,
        )
        self.diagnostics = diagnostics

    def _context_parts(self) -> list[str]:
        return super()._context_parts() + self.diagnostics.summary_parts()


class VertexEmptyResponseError(VertexResponseError):
    """Vertex returned no usable text response."""


class VertexBlockedResponseError(VertexEmptyResponseError):
    """Vertex blocked the prompt or candidate output."""


class VertexResponseParseError(VertexResponseError):
    """Vertex returned text that could not be parsed as the expected JSON."""


class VertexResponseShapeError(VertexResponseError):
    """Vertex returned parsed JSON with an unexpected structure."""
