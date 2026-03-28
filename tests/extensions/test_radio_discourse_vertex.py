from autosub.core.errors import VertexRequestError
from autosub.core.schemas import SubtitleLine
from autosub.extensions.radio_discourse.main import apply_radio_discourse
from autosub.extensions.radio_discourse.classifier import (
    VertexRadioDiscourseClassifier,
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
