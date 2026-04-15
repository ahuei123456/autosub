from __future__ import annotations

import os
from dataclasses import dataclass

LLMProvider = str

SUPPORTED_PROVIDERS: tuple[LLMProvider, ...] = (
    "google-vertex",
    "anthropic-vertex",
    "anthropic",
    "openai",
    "openrouter",
)

PROVIDER_PREFERENCE: tuple[LLMProvider, ...] = (
    "google-vertex",
    "anthropic-vertex",
    "anthropic",
    "openai",
    "openrouter",
)

PROVIDER_ENV_VARS: dict[LLMProvider, str] = {
    "google-vertex": "GOOGLE_CLOUD_PROJECT",
    "anthropic-vertex": "GOOGLE_CLOUD_PROJECT",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

DIRECT_PROVIDER_BY_FAMILY: dict[str, LLMProvider] = {
    "gemini": "google-vertex",
    "claude": "anthropic",
    "openai": "openai",
}

OPENROUTER_VENDOR_PREFIX: dict[str, str] = {
    "gemini": "google",
    "claude": "anthropic",
    "openai": "openai",
}

MODEL_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gemini", "gemini"),
    ("claude", "claude"),
    ("gpt", "openai"),
    ("chatgpt", "openai"),
)


class LLMResolutionError(ValueError):
    """Raised when a model/provider combination cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ResolvedLLMSelection:
    provider: LLMProvider
    model: str | None
    original_model: str | None
    model_family: str | None
    compatible_providers: tuple[LLMProvider, ...]


@dataclass(frozen=True, slots=True)
class ModelClassification:
    original_model: str
    direct_model: str
    model_family: str | None
    compatible_providers: tuple[LLMProvider, ...]
    openrouter_only: bool = False


def resolve_llm_selection(
    *,
    model: str | None,
    provider: str | None,
    default_provider: str = "google-vertex",
) -> ResolvedLLMSelection:
    explicit_provider = provider.strip() if provider else None
    if explicit_provider is not None:
        _validate_provider(explicit_provider)

    if model is None:
        resolved_provider = explicit_provider or default_provider
        _validate_provider(resolved_provider)
        return ResolvedLLMSelection(
            provider=resolved_provider,
            model=None,
            original_model=None,
            model_family=None,
            compatible_providers=(resolved_provider,),
        )

    classification = classify_model(model)
    if classification is None:
        if explicit_provider is None:
            raise LLMResolutionError(
                f"Could not determine a supported provider for model '{model}'. "
                "Use --llm-provider explicitly with a supported model family."
            )
        raise LLMResolutionError(
            f"Model '{model}' is not supported for provider '{explicit_provider}'."
        )

    if explicit_provider is not None:
        if explicit_provider not in classification.compatible_providers:
            compatible_list = ", ".join(classification.compatible_providers)
            raise LLMResolutionError(
                f"Model '{model}' is not compatible with provider '{explicit_provider}'. "
                f"Compatible providers: {compatible_list}."
            )
        _require_provider_credentials(explicit_provider, model)
        resolved_model = _normalize_model_for_provider(
            classification.direct_model,
            explicit_provider,
            classification.model_family,
            original_model=classification.original_model,
        )
        return ResolvedLLMSelection(
            provider=explicit_provider,
            model=resolved_model,
            original_model=classification.original_model,
            model_family=classification.model_family,
            compatible_providers=classification.compatible_providers,
        )

    credentialed_candidates = tuple(
        provider_name
        for provider_name in PROVIDER_PREFERENCE
        if provider_name in classification.compatible_providers
        and provider_has_credentials(provider_name)
    )
    if not credentialed_candidates:
        compatible_list = ", ".join(classification.compatible_providers)
        required_envs = ", ".join(
            PROVIDER_ENV_VARS[provider_name]
            for provider_name in classification.compatible_providers
        )
        raise LLMResolutionError(
            f"Model '{model}' can use these providers: {compatible_list}. "
            f"Set one of these environment variables first: {required_envs}."
        )

    resolved_provider = credentialed_candidates[0]
    resolved_model = _normalize_model_for_provider(
        classification.direct_model,
        resolved_provider,
        classification.model_family,
        original_model=classification.original_model,
    )
    return ResolvedLLMSelection(
        provider=resolved_provider,
        model=resolved_model,
        original_model=classification.original_model,
        model_family=classification.model_family,
        compatible_providers=classification.compatible_providers,
    )


def classify_model(model: str) -> ModelClassification | None:
    normalized = model.strip()
    if not normalized:
        return None

    prefixed = _classify_openrouter_prefixed_model(normalized)
    if prefixed is not None:
        return prefixed

    direct_family = _detect_direct_model_family(normalized)
    if direct_family is None:
        generic_openrouter = _classify_generic_openrouter_model(normalized)
        if generic_openrouter is not None:
            return generic_openrouter
        return None

    direct_provider = DIRECT_PROVIDER_BY_FAMILY[direct_family]
    compatible_providers: tuple[LLMProvider, ...]
    if direct_family == "claude":
        compatible_providers = ("anthropic", "anthropic-vertex", "openrouter")
    else:
        compatible_providers = (direct_provider, "openrouter")
    return ModelClassification(
        original_model=normalized,
        direct_model=normalized,
        model_family=direct_family,
        compatible_providers=compatible_providers,
    )


def provider_has_credentials(provider: str) -> bool:
    _validate_provider(provider)
    env_var = PROVIDER_ENV_VARS[provider]
    return bool(os.environ.get(env_var))


def _require_provider_credentials(provider: str, model: str) -> None:
    if provider_has_credentials(provider):
        return
    env_var = PROVIDER_ENV_VARS[provider]
    raise LLMResolutionError(
        f"Model '{model}' requires provider '{provider}', but {env_var} is not set."
    )


def _validate_provider(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        choices = ", ".join(SUPPORTED_PROVIDERS)
        raise LLMResolutionError(
            f"Unsupported LLM provider '{provider}'. Supported providers: {choices}."
        )


def _detect_direct_model_family(model: str) -> str | None:
    lowered = model.lower()
    for prefix, family in MODEL_FAMILY_PREFIXES:
        if lowered.startswith(prefix):
            return family
    if _looks_like_openai_o_series_model(lowered):
        return "openai"
    return None


def _classify_openrouter_prefixed_model(model: str) -> ModelClassification | None:
    vendor, separator, tail = model.partition("/")
    if not separator or not tail:
        return None

    family = _detect_direct_model_family(tail)
    if family is None:
        return None

    expected_vendor = OPENROUTER_VENDOR_PREFIX.get(family)
    if expected_vendor is None:
        return None

    if expected_vendor != vendor.lower():
        return None

    return ModelClassification(
        original_model=model,
        direct_model=tail,
        model_family=family,
        compatible_providers=("openrouter",),
        openrouter_only=True,
    )


def _classify_generic_openrouter_model(model: str) -> ModelClassification | None:
    vendor, separator, tail = model.partition("/")
    if not separator or not vendor or not tail:
        return None

    return ModelClassification(
        original_model=model,
        direct_model=model,
        model_family=None,
        compatible_providers=("openrouter",),
        openrouter_only=True,
    )


def _normalize_model_for_provider(
    direct_model: str,
    provider: str,
    model_family: str | None,
    *,
    original_model: str,
) -> str:
    if provider != "openrouter":
        return direct_model
    if "/" in original_model:
        return original_model
    if model_family is None:
        return original_model
    vendor_prefix = OPENROUTER_VENDOR_PREFIX.get(model_family)
    return f"{vendor_prefix}/{direct_model}"


def _looks_like_openai_o_series_model(model: str) -> bool:
    return len(model) > 1 and model.startswith("o") and model[1].isdigit()
