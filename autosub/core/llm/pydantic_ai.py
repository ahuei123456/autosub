from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, cast

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.exceptions import ContentFilterError, UnexpectedModelBehavior
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings

from anthropic import AsyncAnthropic

from autosub.core.errors import (
    VertexBlockedResponseError,
    VertexRequestError,
    VertexResponseDiagnostics,
    VertexResponseParseError,
)
from autosub.core.llm.resolver import classify_model

logger = logging.getLogger(__name__)

OutputT = TypeVar("OutputT")
StructuredOutputMode = Literal["native", "tool", "prompted"]


class ReasoningEffort(StrEnum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True, frozen=True)
class LLMModelConfig:
    provider: str
    model: str
    project_id: str | None
    location: str
    temperature: float
    reasoning_effort: ReasoningEffort | None
    reasoning_budget_tokens: int | None
    reasoning_dynamic: bool | None
    provider_options: dict[str, Any] | None


class BaseStructuredLLM:
    _GOOGLE_REASONING_LEVEL = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]
    _GOOGLE_REASONING_BUDGETS: ClassVar[dict[ReasoningEffort, int]] = {
        ReasoningEffort.OFF: 0,
        ReasoningEffort.LOW: 1024,
        ReasoningEffort.MEDIUM: 8192,
        ReasoningEffort.HIGH: 24576,
    }
    _GOOGLE_REASONING_LEVELS: ClassVar[
        dict[ReasoningEffort, Literal["LOW", "HIGH"]]
    ] = {
        ReasoningEffort.LOW: "LOW",
        ReasoningEffort.HIGH: "HIGH",
    }
    _GOOGLE_BUDGET_MODEL_FAMILIES = ("gemini-2.5",)
    _GOOGLE_LEVEL_MODEL_FAMILIES = ("gemini-3",)
    _GOOGLE_MINIMAL_MODEL_PATTERNS = (
        "gemini-3-flash",
        "gemini-3.1-flash-lite",
    )
    _GOOGLE_MEDIUM_MODEL_PATTERNS = (
        "gemini-3-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
    )
    _GOOGLE_MIN_BUDGET_OVERRIDES = (
        ("gemini-2.5-flash-lite", 512),
        ("gemini-2.5-pro", 128),
    )
    _ANTHROPIC_REASONING_BUDGETS: ClassVar[dict[ReasoningEffort, int]] = {
        ReasoningEffort.MINIMAL: 2048,
        ReasoningEffort.LOW: 4096,
        ReasoningEffort.MEDIUM: 16384,
        ReasoningEffort.HIGH: 32768,
    }
    _ANTHROPIC_MIN_BUDGET = 1024
    _ANTHROPIC_DEFAULT_MAX_TOKENS = 65536
    _ANTHROPIC_REASONING_MAX_TOKENS: ClassVar[dict[ReasoningEffort, int]] = {
        ReasoningEffort.MINIMAL: 65536,
        ReasoningEffort.LOW: 65536,
        ReasoningEffort.MEDIUM: 65536,
        ReasoningEffort.HIGH: 65536,
    }
    _TRACE_PART_FIELDS = (
        "content",
        "id",
        "signature",
        "provider_name",
        "provider_details",
    )
    _USAGE_FIELDS = (
        "input_tokens",
        "output_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "requests",
        "tool_calls",
        "details",
    )
    _DIAGNOSTIC_LIST_FIELDS = (
        "prompt_safety_ratings",
        "candidate_finish_reasons",
        "candidate_finish_messages",
        "candidate_token_counts",
        "candidate_safety_ratings",
    )
    _DIAGNOSTIC_SCALAR_FIELDS = (
        "response_id",
        "model_version",
        "prompt_block_reason",
        "prompt_block_reason_message",
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
        "text_preview",
        "http_body_preview",
    )

    def __init__(
        self,
        *,
        project_id: str | None,
        model: str,
        location: str = "global",
        temperature: float = 0.1,
        provider: str = "google-vertex",
        reasoning_effort: ReasoningEffort | str | None = None,
        reasoning_budget_tokens: int | None = None,
        reasoning_dynamic: bool | None = None,
        provider_options: dict[str, Any] | None = None,
        trace_path: Path | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.model = model
        self.location = location
        self.temperature = temperature
        self.provider = provider
        self.reasoning_effort = self._coerce_reasoning_effort(reasoning_effort)
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.reasoning_dynamic = reasoning_dynamic
        self.provider_options = provider_options.copy() if provider_options else None
        self.trace_path = Path(trace_path) if trace_path is not None else None

    def _get_model_config(self) -> LLMModelConfig:
        return LLMModelConfig(
            provider=self.provider,
            model=self.model,
            project_id=self.project_id,
            location=self.location,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            reasoning_budget_tokens=self.reasoning_budget_tokens,
            reasoning_dynamic=self.reasoning_dynamic,
            provider_options=(
                self.provider_options.copy() if self.provider_options else None
            ),
        )

    def _build_model(self) -> Any:
        config = self._get_model_config()

        if config.provider == "google-vertex":
            if not config.project_id:
                raise ValueError(
                    "google-vertex provider requires a Google Cloud project id."
                )

            return GoogleModel(
                config.model,
                provider=GoogleProvider(
                    project=config.project_id,
                    location=config.location,
                ),
                settings=self._build_google_model_settings(config),
            )

        if config.provider == "anthropic":
            return AnthropicModel(
                config.model,
                provider=AnthropicProvider(
                    anthropic_client=AsyncAnthropic(
                        api_key=self._require_anthropic_api_key()
                    )
                ),
                settings=cast(
                    ModelSettings, self._build_anthropic_model_settings(config)
                ),
            )

        if config.provider == "openai":
            return OpenAIResponsesModel(
                config.model,
                provider=OpenAIProvider(api_key=self._require_openai_api_key()),
                settings=cast(ModelSettings, self._build_openai_model_settings(config)),
            )

        if config.provider == "openrouter":
            return OpenRouterModel(
                config.model,
                provider=OpenRouterProvider(api_key=self._require_openrouter_api_key()),
                settings=cast(
                    OpenRouterModelSettings,
                    self._build_openrouter_model_settings(config),
                ),
            )

        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def _build_google_model_settings(
        self, config: LLMModelConfig
    ) -> GoogleModelSettings:
        settings: dict[str, Any] = {
            "temperature": config.temperature,
            "max_tokens": self._ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        thinking_config = self._build_google_thinking_config(config)
        if self.trace_path is not None:
            thinking_config = dict(thinking_config or {})
            thinking_config["include_thoughts"] = True
        if thinking_config is not None:
            settings["google_thinking_config"] = thinking_config
        if config.provider_options:
            settings.update(config.provider_options)
        return cast(GoogleModelSettings, settings)

    def _build_anthropic_model_settings(self, config: LLMModelConfig) -> dict[str, Any]:
        if config.reasoning_dynamic is True:
            raise ValueError("Provider 'anthropic' does not support reasoning_dynamic.")

        settings: dict[str, Any] = {
            "temperature": config.temperature,
            "max_tokens": self._ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        thinking_setting = self._build_anthropic_thinking_setting(config)
        if thinking_setting is not None:
            settings.update(thinking_setting)
        self._normalize_anthropic_temperature(settings)
        if config.provider_options:
            settings.update(config.provider_options)
        self._ensure_anthropic_budget_fits_max_tokens(settings)
        return settings

    def _build_openai_model_settings(self, config: LLMModelConfig) -> dict[str, Any]:
        if config.reasoning_dynamic is True:
            raise ValueError("Provider 'openai' does not support reasoning_dynamic.")

        settings: dict[str, Any] = {"temperature": config.temperature}
        if config.reasoning_effort is not None:
            if config.reasoning_effort == ReasoningEffort.OFF:
                settings["thinking"] = False
            else:
                settings["thinking"] = config.reasoning_effort.value
        if config.provider_options:
            settings.update(config.provider_options)
        return settings

    def _build_openrouter_model_settings(
        self, config: LLMModelConfig
    ) -> dict[str, Any]:
        if config.reasoning_dynamic is True:
            raise ValueError(
                "Provider 'openrouter' does not support reasoning_dynamic."
            )

        settings: dict[str, Any] = {"temperature": config.temperature}
        if (
            config.reasoning_effort is not None
            and config.reasoning_effort != ReasoningEffort.OFF
        ):
            settings["openrouter_reasoning"] = {"effort": config.reasoning_effort.value}
        if config.provider_options:
            settings.update(config.provider_options)
        return settings

    def _build_anthropic_thinking_setting(
        self, config: LLMModelConfig
    ) -> dict[str, Any] | None:
        if config.reasoning_budget_tokens is not None:
            budget_tokens = self._validate_anthropic_thinking_budget(
                config.reasoning_budget_tokens
            )
            return {
                "anthropic_thinking": {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                "max_tokens": max(
                    self._ANTHROPIC_DEFAULT_MAX_TOKENS, budget_tokens * 2
                ),
            }

        if config.reasoning_effort is None:
            return None
        if config.reasoning_effort == ReasoningEffort.OFF:
            return {"thinking": False}

        settings: dict[str, Any] = {"thinking": config.reasoning_effort.value}
        settings["max_tokens"] = self._resolve_anthropic_effort_max_tokens(config)
        return settings

    @classmethod
    def _validate_anthropic_thinking_budget(cls, reasoning_budget_tokens: int) -> int:
        if reasoning_budget_tokens < cls._ANTHROPIC_MIN_BUDGET:
            raise ValueError("anthropic reasoning_budget_tokens must be at least 1024.")
        return reasoning_budget_tokens

    @classmethod
    def _ensure_anthropic_budget_fits_max_tokens(cls, settings: dict[str, Any]) -> None:
        thinking_config = settings.get("anthropic_thinking")
        if not isinstance(thinking_config, dict):
            return

        budget_tokens = thinking_config.get("budget_tokens")
        max_tokens = settings.get("max_tokens")
        if (
            isinstance(budget_tokens, int)
            and isinstance(max_tokens, int)
            and budget_tokens >= max_tokens
        ):
            raise ValueError(
                "anthropic reasoning_budget_tokens must be smaller than max_tokens."
            )

    @staticmethod
    def _resolve_anthropic_effort_budget(config: LLMModelConfig) -> int | None:
        effort = config.reasoning_effort
        if effort is None or effort == ReasoningEffort.OFF:
            return None

        return BaseStructuredLLM._ANTHROPIC_REASONING_BUDGETS[effort]

    @staticmethod
    def _resolve_anthropic_effort_max_tokens(config: LLMModelConfig) -> int:
        effort = config.reasoning_effort
        if effort is None or effort == ReasoningEffort.OFF:
            return BaseStructuredLLM._ANTHROPIC_DEFAULT_MAX_TOKENS

        return BaseStructuredLLM._ANTHROPIC_REASONING_MAX_TOKENS[effort]

    @staticmethod
    def _normalize_anthropic_temperature(settings: dict[str, Any]) -> None:
        thinking_setting = settings.get("thinking")
        anthropic_thinking = settings.get("anthropic_thinking")
        thinking_enabled = thinking_setting not in (None, False) or isinstance(
            anthropic_thinking, dict
        )
        if thinking_enabled:
            settings["temperature"] = 1.0

    def _build_google_thinking_config(
        self, config: LLMModelConfig
    ) -> dict[str, Any] | None:
        if (
            config.reasoning_dynamic is True
            and config.reasoning_budget_tokens is not None
        ):
            raise ValueError(
                "Use either reasoning_dynamic or reasoning_budget_tokens, not both."
            )

        if self._google_uses_budget(config):
            budget = self._resolve_google_thinking_budget(config)
            return None if budget is None else {"thinking_budget": budget}

        level = self._resolve_google_thinking_level(config)
        return None if level is None else {"thinking_level": level}

    def _resolve_google_thinking_budget(self, config: LLMModelConfig) -> int | None:
        if config.reasoning_dynamic is True:
            return -1

        if config.reasoning_budget_tokens is not None:
            return self._validate_google_thinking_budget(config.reasoning_budget_tokens)

        return self._map_reasoning_effort_to_google_budget(
            config.reasoning_effort,
            config.model,
        )

    def _resolve_google_thinking_level(
        self, config: LLMModelConfig
    ) -> Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"] | None:
        if config.reasoning_dynamic is True:
            raise ValueError(
                "Provider 'google-vertex' does not support reasoning_dynamic for this model family."
            )

        if config.reasoning_budget_tokens is not None:
            thinking_level = self._map_google_budget_to_level(
                model_name=config.model,
                reasoning_budget_tokens=config.reasoning_budget_tokens,
            )
        else:
            thinking_level = self._map_google_thinking_level(
                model_name=config.model,
                reasoning_effort=config.reasoning_effort,
            )

        return thinking_level

    @staticmethod
    def _require_anthropic_api_key() -> str:
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("anthropic provider requires ANTHROPIC_API_KEY.")
        return api_key

    @staticmethod
    def _require_openai_api_key() -> str:
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("openai provider requires OPENAI_API_KEY.")
        return api_key

    @staticmethod
    def _require_openrouter_api_key() -> str:
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("openrouter provider requires OPENROUTER_API_KEY.")
        return api_key

    def _build_agent(
        self,
        *,
        system_prompt: str,
        output_type: Any,
        output_name: str,
    ) -> Agent[Any, Any]:
        wrapped_output_type = self._build_agent_output_type(
            output_type=output_type,
            output_name=output_name,
        )
        return Agent(
            self._build_model(),
            system_prompt=system_prompt,
            output_type=wrapped_output_type,
        )

    def _build_agent_output_type(self, *, output_type: Any, output_name: str) -> Any:
        output_mode = self._resolve_structured_output_mode()
        if output_mode == "native":
            return cast(Any, NativeOutput(output_type, name=output_name))
        if output_mode == "prompted":
            return cast(Any, PromptedOutput(output_type, name=output_name))
        return cast(Any, ToolOutput(output_type, name=output_name))

    def _resolve_structured_output_mode(self) -> StructuredOutputMode:
        if self.provider == "openrouter":
            classification = classify_model(self.model)
            if classification is None or classification.model_family is None:
                return "prompted"
            return "tool"
        return "native"

    def _run_structured_output(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        output_type: Any,
        operation_name: str,
        output_name: str,
    ) -> tuple[OutputT, VertexResponseDiagnostics]:
        agent = self._build_agent(
            system_prompt=system_prompt,
            output_type=output_type,
            output_name=output_name,
        )

        try:
            result = agent.run_sync(user_prompt)
        except Exception as exc:
            self._write_trace_entry(
                status="error",
                operation_name=operation_name,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                error=exc,
            )
            raise self._map_run_error(exc, operation_name) from exc

        diagnostics = self._build_response_diagnostics(result)
        logger.debug("%s response diagnostics: %s", operation_name, diagnostics)
        self._write_trace_entry(
            status="success",
            operation_name=operation_name,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            result=result,
            diagnostics=diagnostics,
        )
        return result.output, diagnostics

    def _build_response_diagnostics(
        self, result: AgentRunResult[Any]
    ) -> VertexResponseDiagnostics:
        usage = result.usage()
        response = result.response
        prompt_tokens = getattr(usage, "input_tokens", None) or 0
        candidate_tokens = getattr(usage, "output_tokens", None) or 0
        finish_reason = self._enum_value(response.finish_reason)

        return VertexResponseDiagnostics(
            response_id=response.provider_response_id,
            model_version=response.model_name,
            candidate_finish_reasons=(finish_reason,) if finish_reason else (),
            prompt_token_count=prompt_tokens or None,
            candidates_token_count=candidate_tokens or None,
            total_token_count=(prompt_tokens + candidate_tokens) or None,
            text_preview=self._truncate_preview(self._serialize_output(result.output)),
        )

    def _build_error_diagnostics(
        self,
        *,
        text_preview: str | None = None,
        finish_reason: str | None = None,
    ) -> VertexResponseDiagnostics:
        return VertexResponseDiagnostics(
            model_version=self.model,
            candidate_finish_reasons=(finish_reason,) if finish_reason else (),
            text_preview=self._truncate_preview(text_preview),
        )

    def _map_run_error(
        self,
        exc: Exception,
        operation_name: str,
    ) -> VertexBlockedResponseError | VertexResponseParseError | VertexRequestError:
        if isinstance(exc, ContentFilterError):
            return VertexBlockedResponseError(
                f"{operation_name} returned blocked output: {exc}",
                diagnostics=self._build_error_diagnostics(text_preview=str(exc)),
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            )

        if isinstance(exc, UnexpectedModelBehavior):
            return VertexResponseParseError(
                f"{operation_name} returned invalid structured output: {exc}",
                diagnostics=self._build_error_diagnostics(text_preview=str(exc)),
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            )

        return VertexRequestError(
            f"{operation_name} request to LLM failed: {exc}",
            project_id=self.project_id,
            model=self.model,
            location=self.location,
        )

    @staticmethod
    def _serialize_output(value: Any) -> str | None:
        try:
            return json.dumps(
                BaseStructuredLLM._to_jsonable(value),
                ensure_ascii=False,
            )
        except Exception:
            return repr(value)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [BaseStructuredLLM._to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [BaseStructuredLLM._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {
                key: BaseStructuredLLM._to_jsonable(item) for key, item in value.items()
            }
        return value

    def _write_trace_entry(
        self,
        *,
        status: str,
        operation_name: str,
        user_prompt: str,
        system_prompt: str,
        result: AgentRunResult[Any] | None = None,
        diagnostics: VertexResponseDiagnostics | None = None,
        error: Exception | None = None,
    ) -> None:
        if self.trace_path is None:
            return

        usage = result.usage() if result is not None else None
        response = result.response if result is not None else None
        entry = {
            "status": status,
            "operation_name": operation_name,
            "provider": self.provider,
            "model": self.model,
            "location": self.location,
            "reasoning_effort": self._serialize_reasoning_effort(),
            "reasoning_budget_tokens": self.reasoning_budget_tokens,
            "reasoning_dynamic": self.reasoning_dynamic,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": (
                self._to_jsonable(result.output) if result is not None else None
            ),
            "thinking_parts": self._serialize_thinking_parts(response),
            "response_parts": self._serialize_response_parts(response),
            "usage": self._serialize_usage(usage),
            "response_id": getattr(response, "provider_response_id", None),
            "run_id": getattr(response, "run_id", None),
            "timestamp": self._serialize_timestamp(
                getattr(response, "timestamp", None)
            ),
            "finish_reason": self._enum_value(getattr(response, "finish_reason", None)),
            "diagnostics": self._serialize_diagnostics(diagnostics),
            "error": (
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                if error is not None
                else None
            ),
        }

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")

    @classmethod
    def _serialize_response_parts(cls, response: Any) -> list[dict[str, Any]]:
        return [
            cls._serialize_part(part, include_kind=True)
            for part in getattr(response, "parts", ()) or ()
        ]

    @classmethod
    def _serialize_thinking_parts(cls, response: Any) -> list[dict[str, Any]]:
        return [
            cls._serialize_part(part, include_none=True)
            for part in getattr(response, "parts", ()) or ()
            if isinstance(part, ThinkingPart)
        ]

    @classmethod
    def _serialize_usage(cls, usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        return {field: getattr(usage, field, None) for field in cls._USAGE_FIELDS}

    @classmethod
    def _serialize_diagnostics(
        cls,
        diagnostics: VertexResponseDiagnostics | None,
    ) -> dict[str, Any] | None:
        if diagnostics is None:
            return None
        payload = {
            field: getattr(diagnostics, field)
            for field in cls._DIAGNOSTIC_SCALAR_FIELDS
        }
        payload.update(
            {
                field: list(getattr(diagnostics, field))
                for field in cls._DIAGNOSTIC_LIST_FIELDS
            }
        )
        return payload

    @classmethod
    def _serialize_part(
        cls,
        part: Any,
        *,
        include_kind: bool = False,
        include_none: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if include_kind:
            payload["part_kind"] = getattr(part, "part_kind", type(part).__name__)
        for field in cls._TRACE_PART_FIELDS:
            value = getattr(part, field, None)
            if value is not None or include_none:
                payload[field] = value
        return payload

    def _serialize_reasoning_effort(self) -> str | None:
        if isinstance(self.reasoning_effort, ReasoningEffort):
            return self.reasoning_effort.value
        return self.reasoning_effort

    @staticmethod
    def _serialize_timestamp(value: Any) -> str | None:
        if value is None:
            return None
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return str(isoformat())
        return str(value)

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        return getattr(value, "value", str(value))

    @classmethod
    def _map_google_thinking_level(
        cls,
        *,
        model_name: str,
        reasoning_effort: ReasoningEffort | None,
    ) -> _GOOGLE_REASONING_LEVEL | None:
        if reasoning_effort in {None, ReasoningEffort.OFF}:
            return None
        if reasoning_effort == ReasoningEffort.MINIMAL:
            if cls._google_supports(model_name, cls._GOOGLE_MINIMAL_MODEL_PATTERNS):
                return "MINIMAL"
            raise ValueError(
                f"Provider 'google-vertex' model '{model_name}' does not support reasoning_effort='minimal'."
            )
        if reasoning_effort == ReasoningEffort.MEDIUM:
            if cls._google_supports(model_name, cls._GOOGLE_MEDIUM_MODEL_PATTERNS):
                return "MEDIUM"
            raise ValueError(
                f"Provider 'google-vertex' model '{model_name}' does not support reasoning_effort='medium'."
            )
        return cls._GOOGLE_REASONING_LEVELS.get(reasoning_effort, "HIGH")

    @classmethod
    def _map_reasoning_effort_to_google_budget(
        cls,
        reasoning_effort: ReasoningEffort | None,
        model_name: str,
    ) -> int | None:
        if reasoning_effort is None:
            return None
        if reasoning_effort == ReasoningEffort.MINIMAL:
            return cls._google_min_budget(model_name)
        return cls._GOOGLE_REASONING_BUDGETS.get(reasoning_effort, 24576)

    @classmethod
    def _map_google_budget_to_level(
        cls,
        *,
        model_name: str,
        reasoning_budget_tokens: int,
    ) -> _GOOGLE_REASONING_LEVEL | None:
        budget = cls._validate_google_thinking_budget(reasoning_budget_tokens)
        if budget == 0:
            return None
        if (
            cls._google_supports(model_name, cls._GOOGLE_MINIMAL_MODEL_PATTERNS)
            and budget <= 256
        ):
            return "MINIMAL"
        if budget <= 4096:
            return "LOW"
        if (
            cls._google_supports(model_name, cls._GOOGLE_MEDIUM_MODEL_PATTERNS)
            and budget <= 12288
        ):
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _validate_google_thinking_budget(reasoning_budget_tokens: int) -> int:
        if reasoning_budget_tokens == 0:
            return 0
        if reasoning_budget_tokens == -1:
            return -1
        if 1 <= reasoning_budget_tokens <= 24576:
            return reasoning_budget_tokens
        raise ValueError(
            "google-vertex reasoning_budget_tokens must be 0, -1, or between 1 and 24576."
        )

    @classmethod
    def _google_uses_budget(cls, config: LLMModelConfig) -> bool:
        if cls._google_supports(config.model, cls._GOOGLE_BUDGET_MODEL_FAMILIES):
            return True
        if cls._google_supports(config.model, cls._GOOGLE_LEVEL_MODEL_FAMILIES):
            return False
        return (
            config.reasoning_budget_tokens is not None
            or config.reasoning_dynamic is not None
        )

    @staticmethod
    def _google_supports(model_name: str, patterns: tuple[str, ...]) -> bool:
        lowered = model_name.lower()
        return any(pattern in lowered for pattern in patterns)

    @classmethod
    def _google_min_budget(cls, model_name: str) -> int:
        lowered = model_name.lower()
        for pattern, budget in cls._GOOGLE_MIN_BUDGET_OVERRIDES:
            if pattern in lowered:
                return budget
        return 1

    @staticmethod
    def _coerce_reasoning_effort(
        reasoning_effort: ReasoningEffort | str | None,
    ) -> ReasoningEffort | None:
        if reasoning_effort is None or isinstance(reasoning_effort, ReasoningEffort):
            return reasoning_effort
        return ReasoningEffort(reasoning_effort)

    @staticmethod
    def _truncate_preview(value: str | None, limit: int = 240) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."
