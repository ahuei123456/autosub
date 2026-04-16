import logging
import autosub.cli as cli_module
from types import SimpleNamespace
from typing import cast
from typer.testing import CliRunner

from autosub.cli import app
from autosub.core.llm import ReasoningEffort


runner = CliRunner()


def _write_minimal_ass(path):
    path.write_text(
        "\n".join(
            [
                "[Script Info]",
                "Title: Test",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,こんにちは",
            ]
        ),
        encoding="utf-8",
    )


def test_transcribe_help_includes_backend_options():
    result = runner.invoke(app, ["transcribe", "--help"])

    assert result.exit_code == 0
    assert "--speakers" not in result.output
    assert "--backend" in result.output
    assert "--whisper-model" in result.output


def test_translate_help_uses_chunk_size_only():
    result = runner.invoke(app, ["translate", "--help"])

    assert result.exit_code == 0
    assert "--chunk-size" in result.output
    assert "--no-chunk" not in result.output
    assert "--model" in result.output
    assert "--llm-provider" in result.output
    assert "--llm-reasoning" in result.output
    assert "--llm-location" in result.output


def test_run_help_hides_advanced_translation_knobs():
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--engine" not in result.output
    assert "--no-chunk" not in result.output
    assert "--model" in result.output
    assert "--backend" in result.output
    assert "--whisper-model" in result.output
    assert "--llm-provider" in result.output
    assert "--llm-reasoning" in result.output
    assert "--chunk-size" in result.output
    assert "--start" in result.output
    assert "--end" in result.output


def test_translate_model_infers_provider_and_engine(tmp_path, monkeypatch):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")

    captured: dict[str, object] = {}

    def fake_translate_subtitles(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        cli_module.translate_module, "translate_subtitles", fake_translate_subtitles
    )

    result = runner.invoke(
        app, ["translate", str(input_ass), "--model", "claude-sonnet-4-6"]
    )

    assert result.exit_code == 0
    assert captured["engine"] == "vertex"
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-6"


def test_translate_model_rejects_cloud_v3(tmp_path):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)

    result = runner.invoke(
        app,
        ["translate", str(input_ass), "--engine", "cloud-v3", "--model", "gpt-5-mini"],
    )

    assert result.exit_code == 2
    assert "--model cannot be used with --engine cloud-v3" in result.output


def test_run_model_infers_provider(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module.transcribe_main, "transcribe", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.format_module, "format_subtitles", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.postprocess_module,
        "postprocess_subtitles",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(app, ["run", str(video_path), "--model", "gpt-5-mini"])

    assert result.exit_code == 0
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-5-mini"


def test_run_uses_stage_grouped_profile_settings(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    format_kwargs: dict[str, object] = {}
    translate_kwargs: dict[str, object] = {}
    postprocess_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module.transcribe_main, "transcribe", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.format_module,
        "format_subtitles",
        lambda *args, **kwargs: format_kwargs.update(kwargs),
    )
    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: translate_kwargs.update(kwargs),
    )
    monkeypatch.setattr(
        cli_module.postprocess_module,
        "postprocess_subtitles",
        lambda *args, **kwargs: postprocess_kwargs.update(kwargs),
    )
    monkeypatch.setattr(
        cli_module,
        "load_unified_profile",
        lambda profile: {
            "transcribe": {"vocab": ["鈴原希実"]},
            "format": {
                "min_duration_ms": 900,
                "extensions": {"radio_discourse": {"enabled": True}},
                "replacements": {"鈴原のぞみ": "鈴原希実"},
            },
            "translate": {
                "prompt": ["Keep honorifics consistent."],
                "glossary": {"鈴原希実": "Suzuhara Nozomi"},
            },
            "postprocess": {"extensions": {"radio_discourse": {"enabled": False}}},
        },
    )

    result = runner.invoke(app, ["run", str(video_path), "--profile", "test_profile"])

    assert result.exit_code == 0
    assert format_kwargs["timing_config"] == {"min_duration_ms": 900}
    assert format_kwargs["extensions_config"] == {"radio_discourse": {"enabled": True}}
    assert format_kwargs["normalizer_config"] == {
        "engine": "exact",
        "replacements": {"鈴原のぞみ": "鈴原希実"},
    }
    assert (
        cast(str, translate_kwargs["system_prompt"])
        == 'Keep honorifics consistent.\n\nGlossary (Always translate these exact phrases):\n- "鈴原希実" -> "Suzuhara Nozomi"\n'
    )
    assert postprocess_kwargs["extensions_config"] == {
        "radio_discourse": {"enabled": False}
    }


def test_translate_model_falls_back_to_openrouter_when_only_openrouter_key_exists(
    tmp_path, monkeypatch
):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    captured: dict[str, object] = {}

    def fake_translate_subtitles(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        cli_module.translate_module, "translate_subtitles", fake_translate_subtitles
    )

    result = runner.invoke(app, ["translate", str(input_ass), "--model", "gpt-5-mini"])

    assert result.exit_code == 0
    assert captured["engine"] == "vertex"
    assert captured["provider"] == "openrouter"
    assert captured["model"] == "openai/gpt-5-mini"


def test_translate_accepts_openrouter_native_model_id(tmp_path, monkeypatch):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    captured: dict[str, object] = {}

    def fake_translate_subtitles(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        cli_module.translate_module, "translate_subtitles", fake_translate_subtitles
    )

    result = runner.invoke(
        app,
        ["translate", str(input_ass), "--model", "qwen/qwen3.6-plus:free"],
    )

    assert result.exit_code == 0
    assert captured["engine"] == "vertex"
    assert captured["provider"] == "openrouter"
    assert captured["model"] == "qwen/qwen3.6-plus:free"


def test_extract_format_profile_config_rejects_llm_normalizer_with_replacements():
    profile_data = {
        "format": {
            "normalizer": {"engine": "llm", "terms": [{"value": "鈴原希実"}]},
            "replacements": {"鈴原のぞみ": "鈴原希実"},
        }
    }

    try:
        cli_module._extract_format_profile_config(profile_data)
    except ValueError as exc:
        assert "[format.replacements] cannot be combined" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting normalizer config.")


def test_transcribe_supports_multiple_ranges(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_transcribe(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(words=[])

    monkeypatch.setattr(cli_module.transcribe_main, "transcribe", fake_transcribe)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(video_path),
            "--start",
            "0",
            "--start",
            "15",
            "--end",
            "5",
            "--end",
            "20",
        ],
    )

    assert result.exit_code == 0
    assert captured["time_ranges"] == [("0", "5"), ("15", "20")]


def test_transcribe_supports_whisperx_backend_options(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_transcribe(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(words=[])

    monkeypatch.setattr(cli_module.transcribe_main, "transcribe", fake_transcribe)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(video_path),
            "--backend",
            "whisperx",
            "--whisper-model",
            "medium",
            "--whisper-device",
            "cuda",
            "--whisper-compute-type",
            "float16",
            "--whisper-batch-size",
            "8",
            "--whisper-diarize",
            "--whisper-hf-token",
            "hf-token",
        ],
    )

    assert result.exit_code == 0
    assert captured["transcription_backend"] == "whisperx"
    assert captured["whisper_model"] == "medium"
    assert captured["whisper_device"] == "cuda"
    assert captured["whisper_compute_type"] == "float16"
    assert captured["whisper_batch_size"] == 8
    assert captured["whisper_diarize"] is True
    assert captured["whisper_hf_token"] == "hf-token"


def test_transcribe_logs_exception_type_for_empty_timeout_error(
    tmp_path, monkeypatch, caplog
):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")

    def fail_transcribe(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(cli_module.transcribe_main, "transcribe", fail_transcribe)

    with caplog.at_level(logging.ERROR):
        result = runner.invoke(app, ["transcribe", str(video_path)])

    assert result.exit_code == 1
    assert any(
        "Error during transcription (TimeoutError)" in record.getMessage()
        for record in caplog.records
    )


def test_transcribe_rejects_mismatched_multiple_ranges(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(video_path),
            "--start",
            "0",
            "--start",
            "15",
            "--end",
            "5",
        ],
    )

    assert result.exit_code == 2
    assert "the number of starts and ends must match" in result.output


def test_translate_uses_default_config_when_flags_are_absent(tmp_path, monkeypatch):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                "[translate]",
                'llm_provider = "openai"',
                'reasoning_effort = "low"',
                "bilingual = true",
                "chunk_size = 12",
                'target = "fr"',
                'llm_location = "us-central1"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(app, ["translate", str(input_ass)])

    assert result.exit_code == 0
    assert captured["provider"] == "openai"
    assert captured["reasoning_effort"] == ReasoningEffort.LOW
    assert captured["bilingual"] is True
    assert captured["chunk_size"] == 12
    assert captured["target_lang"] == "fr"
    assert captured["location"] == "us-central1"


def test_translate_command_line_flags_override_default_config(tmp_path, monkeypatch):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                "[translate]",
                'llm_provider = "openai"',
                'reasoning_effort = "low"',
                "bilingual = true",
                "chunk_size = 12",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(
        app,
        [
            "translate",
            str(input_ass),
            "--llm-provider",
            "anthropic",
            "--replace",
            "--chunk-size",
            "4",
            "--vertex-reasoning-effort",
            "high",
        ],
    )

    assert result.exit_code == 0
    assert captured["provider"] == "anthropic"
    assert captured["bilingual"] is False
    assert captured["chunk_size"] == 4
    assert captured["reasoning_effort"] == ReasoningEffort.HIGH


def test_no_config_ignores_default_config_file(tmp_path, monkeypatch):
    input_ass = tmp_path / "original.ass"
    _write_minimal_ass(input_ass)
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                "[translate]",
                'llm_provider = "openai"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(app, ["--no-config", "translate", str(input_ass)])

    assert result.exit_code == 0
    assert captured["provider"] == "google-vertex"


def test_run_inherits_stage_defaults_without_run_section(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                "[transcribe]",
                'language = "en-US"',
                'vocab = ["idol", "seiyuu"]',
                'backend = "whisperx"',
                'whisper_model = "medium"',
                'whisper_device = "cuda"',
                'whisper_compute_type = "float16"',
                "whisper_batch_size = 8",
                "whisper_diarize = true",
                'whisper_hf_token = "hf-token"',
                'start = "00:01:00"',
                'end = "00:02:00"',
                "",
                "[translate]",
                'llm_provider = "openai"',
                'model = "gpt-5-mini"',
                'reasoning_effort = "low"',
                "bilingual = true",
                "chunk_size = 7",
                'target = "fr"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    transcribe_args = None
    transcribe_kwargs = None
    captured_translate: dict[str, object] = {}

    def fake_transcribe(*args, **kwargs):
        nonlocal transcribe_args, transcribe_kwargs
        transcribe_args = args
        transcribe_kwargs = kwargs

    monkeypatch.setattr(
        cli_module.transcribe_main,
        "transcribe",
        fake_transcribe,
    )
    monkeypatch.setattr(
        cli_module.format_module, "format_subtitles", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.postprocess_module,
        "postprocess_subtitles",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: captured_translate.update(kwargs),
    )

    result = runner.invoke(app, ["run", str(video_path)])

    assert result.exit_code == 0
    assert transcribe_args is not None
    assert transcribe_kwargs is not None
    assert transcribe_args[2] == "en-US"
    assert transcribe_args[3] == ["idol", "seiyuu"]
    assert transcribe_kwargs["transcription_backend"] == "whisperx"
    assert transcribe_kwargs["whisper_model"] == "medium"
    assert transcribe_kwargs["whisper_device"] == "cuda"
    assert transcribe_kwargs["whisper_compute_type"] == "float16"
    assert transcribe_kwargs["whisper_batch_size"] == 8
    assert transcribe_kwargs["whisper_diarize"] is True
    assert transcribe_kwargs["whisper_hf_token"] == "hf-token"
    assert transcribe_kwargs["time_ranges"] == [("00:01:00", "00:02:00")]
    assert captured_translate["provider"] == "openai"
    assert captured_translate["model"] == "gpt-5-mini"
    assert captured_translate["reasoning_effort"] == ReasoningEffort.LOW
    assert captured_translate["bilingual"] is True
    assert captured_translate["chunk_size"] == 7
    assert captured_translate["target_lang"] == "fr"


def test_run_supports_multiple_transcription_ranges(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")

    captured_transcribe: dict[str, object] = {}

    def fake_transcribe(*args, **kwargs):
        captured_transcribe.update(kwargs)

    monkeypatch.setattr(cli_module.transcribe_main, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        cli_module.format_module, "format_subtitles", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.postprocess_module,
        "postprocess_subtitles",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: None,
    )

    result = runner.invoke(
        app,
        [
            "run",
            str(video_path),
            "--start",
            "0",
            "--start",
            "15",
            "--end",
            "5",
            "--end",
            "20",
        ],
    )

    assert result.exit_code == 0
    assert captured_transcribe["time_ranges"] == [("0", "5"), ("15", "20")]


def test_run_supports_whisperx_backend_options(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")

    captured_transcribe: dict[str, object] = {}

    def fake_transcribe(*args, **kwargs):
        captured_transcribe.update(kwargs)

    monkeypatch.setattr(cli_module.transcribe_main, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        cli_module.format_module, "format_subtitles", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_module.postprocess_module,
        "postprocess_subtitles",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        cli_module.translate_module,
        "translate_subtitles",
        lambda *args, **kwargs: None,
    )

    result = runner.invoke(
        app,
        [
            "run",
            str(video_path),
            "--backend",
            "whisperx",
            "--whisper-model",
            "medium",
            "--whisper-device",
            "cuda",
            "--whisper-compute-type",
            "float16",
            "--whisper-batch-size",
            "8",
            "--whisper-diarize",
            "--whisper-hf-token",
            "hf-token",
        ],
    )

    assert result.exit_code == 0
    assert captured_transcribe["transcription_backend"] == "whisperx"
    assert captured_transcribe["whisper_model"] == "medium"
    assert captured_transcribe["whisper_device"] == "cuda"
    assert captured_transcribe["whisper_compute_type"] == "float16"
    assert captured_transcribe["whisper_batch_size"] == 8
    assert captured_transcribe["whisper_diarize"] is True
    assert captured_transcribe["whisper_hf_token"] == "hf-token"
