import pytest

from autosub.core.errors import VertexRequestError
from autosub.core.schemas import SubtitleLine
from autosub.extensions.radio_discourse.main import apply_radio_discourse
from autosub.extensions.radio_discourse.classifier import (
    VertexRadioDiscourseClassifier,
    classify_roles_with_vertex,
    _build_windows_for_config,
)


def test_vertex_radio_discourse_prompt_describes_roles():
    classifier = VertexRadioDiscourseClassifier(project_id="test-project")

    instruction = classifier._get_system_instruction(1)

    assert "host_meta" in instruction
    assert "listener_mail" in instruction
    assert "Do not rewrite or normalize the text" in instruction
    assert "のちゃん、の番は?" in instruction
    assert "ノンバーワン。" in instruction


def test_apply_radio_discourse_hybrid_uses_vertex_roles(monkeypatch):
    lines = [
        SubtitleLine(
            text="ではメッセージをご紹介していきます。", start_time=0.0, end_time=1.0
        ),
        SubtitleLine(text="メールを送るのは初めてです。", start_time=1.0, end_time=2.0),
        SubtitleLine(text="おお、嬉しい。", start_time=2.0, end_time=3.0),
    ]

    def fake_classify(processed_lines, fallback_roles, config):
        assert config["engine"] == "hybrid"
        return ["host_meta", "listener_mail", "host"]

    monkeypatch.setattr(
        "autosub.extensions.radio_discourse.main.classify_roles_with_vertex",
        fake_classify,
    )

    result = apply_radio_discourse(lines, {"enabled": True, "engine": "hybrid"})

    assert [line.role for line in result] == ["host_meta", "listener_mail", "host"]


def test_vertex_radio_discourse_full_script_scope_uses_single_window():
    lines = [
        SubtitleLine(
            text=f"line {index}", start_time=float(index), end_time=float(index + 1)
        )
        for index in range(25)
    ]

    windows = _build_windows_for_config(lines, {"scope": "full_script"})

    assert len(windows) == 1
    assert len(windows[0]) == 25
    assert windows[0][0][0] == 0
    assert windows[0][-1][0] == 24


def test_classify_roles_with_vertex_defaults_to_flash_lite(monkeypatch):
    captured: dict[str, str] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["model"] = self.model
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"project_id": "test-project"},
    )

    assert result == ["host"]
    assert captured["model"] == "gemini-3.1-flash-lite-preview"


def test_classify_roles_with_vertex_defaults_to_haiku_for_anthropic(monkeypatch):
    captured: dict[str, str] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["model"] = self.model
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"provider": "anthropic"},
    )

    assert result == ["host"]
    assert captured["model"] == "claude-haiku-4-5"


def test_classify_roles_with_vertex_defaults_to_haiku_for_anthropic_vertex(
    monkeypatch,
):
    captured: dict[str, str] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["model"] = self.model
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"provider": "anthropic-vertex", "project_id": "test-project"},
    )

    assert result == ["host"]
    assert captured["model"] == "claude-haiku-4-5"


def test_classify_roles_with_vertex_defaults_to_gpt_5_mini_for_openai(monkeypatch):
    captured: dict[str, str] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["model"] = self.model
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"provider": "openai"},
    )

    assert result == ["host"]
    assert captured["model"] == "gpt-5-mini"


def test_classify_roles_with_vertex_passes_reasoning_effort(monkeypatch):
    captured: dict[str, str | None] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["reasoning_effort"] = self.reasoning_effort
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"project_id": "test-project", "reasoning_effort": "high"},
    )

    assert result == ["host"]
    assert captured["reasoning_effort"] == "high"


def test_classify_roles_with_vertex_passes_reasoning_budget(monkeypatch):
    captured: dict[str, int | None] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"project_id": "test-project", "reasoning_budget_tokens": 2048},
    )

    assert result == ["host"]
    assert captured["reasoning_budget_tokens"] == 2048


def test_classify_roles_with_vertex_allows_anthropic_without_project_id(monkeypatch):
    captured: dict[str, str | None] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["provider"] = self.provider
        captured["project_id"] = self.project_id
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"provider": "anthropic"},
    )

    assert result == ["host"]
    assert captured["provider"] == "anthropic"
    assert captured["project_id"] is None


def test_classify_roles_with_vertex_requires_project_id_for_anthropic_vertex():
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    with pytest.raises(ValueError, match="requires a Google Cloud project id"):
        classify_roles_with_vertex(
            lines,
            fallback_roles=[None],
            config={"provider": "anthropic-vertex"},
        )


def test_classify_roles_with_vertex_allows_openai_without_project_id(monkeypatch):
    captured: dict[str, str | None] = {}
    lines = [SubtitleLine(text="line 0", start_time=0.0, end_time=1.0)]

    def fake_classify_window(self, window):
        captured["provider"] = self.provider
        captured["project_id"] = self.project_id
        return {0: "host"}

    monkeypatch.setattr(
        VertexRadioDiscourseClassifier,
        "classify_window",
        fake_classify_window,
    )

    result = classify_roles_with_vertex(
        lines,
        fallback_roles=[None],
        config={"provider": "openai"},
    )

    assert result == ["host"]
    assert captured["provider"] == "openai"
    assert captured["project_id"] is None


def test_apply_radio_discourse_hybrid_falls_back_to_rules_on_vertex_failure(
    monkeypatch,
):
    lines = [
        SubtitleLine(
            text="いろんなコーデに合わせられるのでおすすめですといただきました。",
            start_time=0.0,
            end_time=2.0,
        )
    ]

    def fake_classify(processed_lines, fallback_roles, config):
        raise VertexRequestError("simulated vertex failure")

    monkeypatch.setattr(
        "autosub.extensions.radio_discourse.main.classify_roles_with_vertex",
        fake_classify,
    )

    result = apply_radio_discourse(lines, {"enabled": True, "engine": "hybrid"})

    assert [line.text for line in result] == [
        "いろんなコーデに合わせられるのでおすすめです。",
        "といただきました。",
    ]
    assert [line.role for line in result] == ["listener_mail", "host_meta"]
