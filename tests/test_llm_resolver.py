import pytest

from autosub.core.llm import LLMResolutionError, resolve_llm_selection


def _clear_llm_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_resolver_prefers_openai_over_openrouter_for_gpt_family(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(model="gpt-5-mini", provider=None)

    assert selection.provider == "openai"
    assert selection.model == "gpt-5-mini"
    assert selection.model_family == "openai"


def test_resolver_falls_back_to_openrouter_for_gpt_family(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(model="gpt-5-mini", provider=None)

    assert selection.provider == "openrouter"
    assert selection.model == "openai/gpt-5-mini"
    assert selection.model_family == "openai"


def test_resolver_prefers_anthropic_over_openrouter_for_claude_family(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(model="claude-sonnet-4-6", provider=None)

    assert selection.provider == "anthropic"
    assert selection.model == "claude-sonnet-4-6"
    assert selection.model_family == "claude"


def test_resolver_falls_back_to_anthropic_vertex_for_claude_family(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(model="claude-sonnet-4-6", provider=None)

    assert selection.provider == "anthropic-vertex"
    assert selection.model == "claude-sonnet-4-6"
    assert selection.model_family == "claude"


def test_resolver_prefers_vertex_over_openrouter_for_gemini_family(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(model="gemini-3-flash-preview", provider=None)

    assert selection.provider == "google-vertex"
    assert selection.model == "gemini-3-flash-preview"
    assert selection.model_family == "gemini"


def test_resolver_respects_explicit_openrouter_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(model="claude-sonnet-4-6", provider="openrouter")

    assert selection.provider == "openrouter"
    assert selection.model == "anthropic/claude-sonnet-4-6"
    assert selection.model_family == "claude"


def test_resolver_supports_explicit_anthropic_vertex_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    selection = resolve_llm_selection(
        model="claude-sonnet-4-6", provider="anthropic-vertex"
    )

    assert selection.provider == "anthropic-vertex"
    assert selection.model == "claude-sonnet-4-6"
    assert selection.model_family == "claude"


def test_resolver_supports_openrouter_prefixed_models(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(
        model="openai/gpt-5-mini",
        provider="openrouter",
    )

    assert selection.provider == "openrouter"
    assert selection.model == "openai/gpt-5-mini"
    assert selection.compatible_providers == ("openrouter",)


def test_resolver_rejects_incompatible_explicit_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    with pytest.raises(
        LLMResolutionError, match="not compatible with provider 'openai'"
    ):
        resolve_llm_selection(model="claude-sonnet-4-6", provider="openai")


def test_resolver_requires_credentials_for_explicit_provider(monkeypatch):
    _clear_llm_env(monkeypatch)

    with pytest.raises(LLMResolutionError, match="OPENROUTER_API_KEY is not set"):
        resolve_llm_selection(model="gpt-5-mini", provider="openrouter")


def test_resolver_requires_google_project_for_explicit_anthropic_vertex_provider(
    monkeypatch,
):
    _clear_llm_env(monkeypatch)

    with pytest.raises(LLMResolutionError, match="GOOGLE_CLOUD_PROJECT is not set"):
        resolve_llm_selection(model="claude-sonnet-4-6", provider="anthropic-vertex")


def test_resolver_accepts_explicit_openrouter_native_model(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(
        model="qwen/qwen3.6-plus:free",
        provider="openrouter",
    )

    assert selection.provider == "openrouter"
    assert selection.model == "qwen/qwen3.6-plus:free"
    assert selection.model_family is None
    assert selection.compatible_providers == ("openrouter",)


def test_resolver_auto_selects_openrouter_for_vendor_prefixed_openrouter_model(
    monkeypatch,
):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(
        model="qwen/qwen3.6-plus:free",
        provider=None,
    )

    assert selection.provider == "openrouter"
    assert selection.model == "qwen/qwen3.6-plus:free"
    assert selection.model_family is None
    assert selection.compatible_providers == ("openrouter",)


def test_resolver_treats_arbitrary_vendor_prefixed_models_as_openrouter(
    monkeypatch,
):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    selection = resolve_llm_selection(
        model="openai/claude-sonnet-4-6",
        provider="openrouter",
    )

    assert selection.provider == "openrouter"
    assert selection.model == "openai/claude-sonnet-4-6"
    assert selection.model_family is None
