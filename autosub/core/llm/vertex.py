from __future__ import annotations

import json
import logging
import time
from typing import Any

from autosub.core.errors import (
    VertexBlockedResponseError,
    VertexEmptyResponseError,
    VertexRequestError,
    VertexResponseDiagnostics,
    VertexResponseParseError,
)
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class BaseVertexLLM:
    def __init__(
        self,
        *,
        project_id: str,
        model: str,
        location: str = "global",
        temperature: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.model = model
        self.location = location
        self.temperature = temperature

    def _get_client(self) -> genai.Client:
        return genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def _generate_structured_json(
        self,
        *,
        contents: str,
        system_instruction: str,
        response_schema: Any,
        operation_name: str,
    ) -> tuple[Any, VertexResponseDiagnostics]:
        client = self._get_client()
        t0 = time.monotonic()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=self.temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=8192),
                ),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            raise VertexRequestError(
                f"{operation_name} request to Vertex failed after {elapsed:.1f}s: {exc}",
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc
        finally:
            try:
                client.close()
            except Exception:
                logger.debug("Failed to close Vertex client cleanly.", exc_info=True)

        diagnostics = self._build_response_diagnostics(response)
        logger.debug("%s response diagnostics: %s", operation_name, diagnostics)

        # Warn when finish_reason is not STOP (e.g. MAX_TOKENS, OTHER)
        non_stop = [r for r in diagnostics.candidate_finish_reasons if r != "STOP"]
        if non_stop:
            logger.warning(
                "%s finished with non-STOP reason(s): %s (diagnostics: %s)",
                operation_name,
                ", ".join(non_stop),
                "; ".join(diagnostics.summary_parts()),
            )

        if not response.text:
            error_class = (
                VertexBlockedResponseError
                if self._is_blocked_response(diagnostics)
                else VertexEmptyResponseError
            )
            finish_info = (
                f" finish_reason={', '.join(diagnostics.candidate_finish_reasons)}"
                if diagnostics.candidate_finish_reasons
                else ""
            )
            raise error_class(
                f"{operation_name} returned no text response.{finish_info}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            )

        try:
            return json.loads(response.text), diagnostics
        except json.JSONDecodeError as exc:
            raise VertexResponseParseError(
                f"{operation_name} returned invalid JSON: {exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc

    def _build_response_diagnostics(
        self, response: types.GenerateContentResponse
    ) -> VertexResponseDiagnostics:
        candidate_finish_reasons: list[str] = []
        candidate_finish_messages: list[str] = []
        candidate_token_counts: list[int] = []
        candidate_safety_ratings: list[str] = []

        for candidate in response.candidates or []:
            finish_reason = self._enum_value(candidate.finish_reason)
            if finish_reason:
                candidate_finish_reasons.append(finish_reason)

            if candidate.finish_message:
                candidate_finish_messages.append(candidate.finish_message)

            if candidate.token_count is not None:
                candidate_token_counts.append(candidate.token_count)

            rating_summary = self._summarize_safety_ratings(candidate.safety_ratings)
            if rating_summary:
                candidate_index = (
                    candidate.index
                    if candidate.index is not None
                    else len(candidate_safety_ratings)
                )
                candidate_safety_ratings.append(
                    f"candidate[{candidate_index}]={rating_summary}"
                )

        prompt_feedback = response.prompt_feedback
        usage_metadata = response.usage_metadata
        sdk_http_response = response.sdk_http_response
        prompt_safety_summary = self._summarize_safety_ratings(
            prompt_feedback.safety_ratings if prompt_feedback else None
        )

        return VertexResponseDiagnostics(
            response_id=response.response_id,
            model_version=response.model_version,
            prompt_block_reason=self._enum_value(
                prompt_feedback.block_reason if prompt_feedback else None
            ),
            prompt_block_reason_message=(
                prompt_feedback.block_reason_message if prompt_feedback else None
            ),
            prompt_safety_ratings=(
                tuple(prompt_safety_summary.split("; "))
                if prompt_safety_summary
                else ()
            ),
            candidate_finish_reasons=tuple(candidate_finish_reasons),
            candidate_finish_messages=tuple(candidate_finish_messages),
            candidate_token_counts=tuple(candidate_token_counts),
            candidate_safety_ratings=tuple(candidate_safety_ratings),
            prompt_token_count=(
                usage_metadata.prompt_token_count if usage_metadata else None
            ),
            candidates_token_count=(
                usage_metadata.candidates_token_count if usage_metadata else None
            ),
            total_token_count=(
                usage_metadata.total_token_count if usage_metadata else None
            ),
            thoughts_token_count=(
                usage_metadata.thoughts_token_count if usage_metadata else None
            ),
            text_preview=self._truncate_preview(response.text),
            http_body_preview=self._truncate_preview(
                sdk_http_response.body if sdk_http_response else None
            ),
        )

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        return getattr(value, "value", str(value))

    def _summarize_safety_ratings(
        self, ratings: list[types.SafetyRating] | None
    ) -> str:
        summaries: list[str] = []
        for rating in ratings or []:
            category = self._enum_value(rating.category)
            probability = self._enum_value(rating.probability)
            parts = [part for part in (category, probability) if part]
            if rating.blocked is not None:
                parts.append(f"blocked={rating.blocked}")
            if parts:
                summaries.append(",".join(parts))
        return "; ".join(summaries)

    @staticmethod
    def _truncate_preview(value: str | None, limit: int = 240) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    @staticmethod
    def _is_blocked_response(diagnostics: VertexResponseDiagnostics) -> bool:
        if diagnostics.prompt_block_reason:
            return True

        blocked_finish_reasons = {
            "SAFETY",
            "RECITATION",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
            "NO_IMAGE",
        }
        return any(
            reason in blocked_finish_reasons
            for reason in diagnostics.candidate_finish_reasons
        )
