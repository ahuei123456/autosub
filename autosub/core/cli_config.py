from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from click.core import ParameterSource

from autosub.core.llm import ReasoningEffort


class CLIConfigError(ValueError):
    """Raised when config.toml contains invalid CLI defaults."""


@dataclass(frozen=True)
class OptionSpec:
    aliases: tuple[str, ...]
    converter: Callable[[Any], Any]


def _as_path(value: Any) -> Path:
    if not isinstance(value, str):
        raise CLIConfigError("expected a string path")
    return Path(value)


def _as_str(value: Any) -> str:
    if not isinstance(value, str):
        raise CLIConfigError("expected a string")
    return value


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise CLIConfigError("expected an array of strings")
    if not all(isinstance(item, str) for item in value):
        raise CLIConfigError("expected an array of strings")
    return value


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CLIConfigError("expected a number")
    return float(value)


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CLIConfigError("expected an integer")
    return value


def _as_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise CLIConfigError("expected true or false")
    return value


def _as_reasoning_effort(value: Any) -> ReasoningEffort:
    if not isinstance(value, str):
        raise CLIConfigError("expected one of: off, minimal, low, medium, high")
    try:
        return ReasoningEffort(value)
    except ValueError as exc:
        raise CLIConfigError(
            "expected one of: off, minimal, low, medium, high"
        ) from exc


COMMAND_OPTION_SPECS: dict[str, dict[str, OptionSpec]] = {
    "transcribe": {
        "output": OptionSpec(("out", "output"), _as_path),
        "language": OptionSpec(("language",), _as_str),
        "vocab": OptionSpec(("vocab",), _as_str_list),
        "profile": OptionSpec(("profile",), _as_str),
        "transcription_backend": OptionSpec(
            ("transcription_backend", "backend"), _as_str
        ),
        "whisper_model": OptionSpec(("whisper_model",), _as_str),
        "whisper_device": OptionSpec(("whisper_device",), _as_str),
        "whisper_compute_type": OptionSpec(("whisper_compute_type",), _as_str),
        "whisper_batch_size": OptionSpec(("whisper_batch_size",), _as_int),
        "whisper_diarize": OptionSpec(("whisper_diarize",), _as_bool),
        "whisper_hf_token": OptionSpec(("whisper_hf_token",), _as_str),
        "start": OptionSpec(("start",), _as_str),
        "end": OptionSpec(("end",), _as_str),
    },
    "format": {
        "out": OptionSpec(("out",), _as_path),
        "json_out": OptionSpec(("json_out",), _as_path),
        "keyframes": OptionSpec(("keyframes",), _as_path),
        "fps": OptionSpec(("fps",), _as_float),
        "profile": OptionSpec(("profile",), _as_str),
    },
    "translate": {
        "out": OptionSpec(("out",), _as_path),
        "ass_out": OptionSpec(("ass_out",), _as_path),
        "engine": OptionSpec(("engine",), _as_str),
        "prompt": OptionSpec(("prompt",), _as_str),
        "profile": OptionSpec(("profile",), _as_str),
        "target_lang": OptionSpec(("target", "target_lang"), _as_str),
        "source_lang": OptionSpec(("source", "source_lang"), _as_str),
        "vertex_model": OptionSpec(("model", "llm_model", "vertex_model"), _as_str),
        "vertex_location": OptionSpec(
            ("llm_location", "vertex_location", "location"), _as_str
        ),
        "llm_provider": OptionSpec(("llm_provider", "provider"), _as_str),
        "vertex_reasoning_effort": OptionSpec(
            ("reasoning_effort", "llm_reasoning_effort", "vertex_reasoning_effort"),
            _as_reasoning_effort,
        ),
        "vertex_reasoning_budget": OptionSpec(
            (
                "reasoning_budget",
                "llm_reasoning_budget",
                "vertex_reasoning_budget",
                "reasoning_budget_tokens",
            ),
            _as_int,
        ),
        "vertex_reasoning_dynamic": OptionSpec(
            ("reasoning_dynamic", "llm_reasoning_dynamic", "vertex_reasoning_dynamic"),
            _as_bool,
        ),
        "bilingual": OptionSpec(("bilingual",), _as_bool),
        "chunk_size": OptionSpec(("chunk_size",), _as_int),
    },
    "postprocess": {
        "out": OptionSpec(("out",), _as_path),
        "ass_out": OptionSpec(("ass_out",), _as_path),
        "profile": OptionSpec(("profile",), _as_str),
        "bilingual": OptionSpec(("bilingual",), _as_bool),
    },
    "run": {
        "out_dir": OptionSpec(("out_dir", "output_dir"), _as_path),
        "language": OptionSpec(("language",), _as_str),
        "profile": OptionSpec(("profile",), _as_str),
        "vocab": OptionSpec(("vocab",), _as_str_list),
        "transcription_backend": OptionSpec(
            ("transcription_backend", "backend"), _as_str
        ),
        "whisper_model": OptionSpec(("whisper_model",), _as_str),
        "whisper_device": OptionSpec(("whisper_device",), _as_str),
        "whisper_compute_type": OptionSpec(("whisper_compute_type",), _as_str),
        "whisper_batch_size": OptionSpec(("whisper_batch_size",), _as_int),
        "whisper_diarize": OptionSpec(("whisper_diarize",), _as_bool),
        "whisper_hf_token": OptionSpec(("whisper_hf_token",), _as_str),
        "prompt": OptionSpec(("prompt",), _as_str),
        "target_lang": OptionSpec(("target", "target_lang"), _as_str),
        "source_lang": OptionSpec(("source", "source_lang"), _as_str),
        "model": OptionSpec(("model", "llm_model"), _as_str),
        "vertex_reasoning_effort": OptionSpec(
            ("reasoning_effort", "llm_reasoning_effort", "vertex_reasoning_effort"),
            _as_reasoning_effort,
        ),
        "llm_provider": OptionSpec(("llm_provider", "provider"), _as_str),
        "bilingual": OptionSpec(("bilingual",), _as_bool),
        "keyframes": OptionSpec(("keyframes",), _as_path),
        "extract_keyframes": OptionSpec(("extract_keyframes",), _as_bool),
        "start": OptionSpec(("start",), _as_str),
        "end": OptionSpec(("end",), _as_str),
        "chunk_size": OptionSpec(("chunk_size",), _as_int),
    },
}

RUN_SECTION_FALLBACKS: dict[str, tuple[tuple[str, str], ...]] = {
    "language": (("transcribe", "language"),),
    "profile": (
        ("transcribe", "profile"),
        ("format", "profile"),
        ("translate", "profile"),
        ("postprocess", "profile"),
    ),
    "vocab": (("transcribe", "vocab"),),
    "transcription_backend": (("transcribe", "transcription_backend"),),
    "whisper_model": (("transcribe", "whisper_model"),),
    "whisper_device": (("transcribe", "whisper_device"),),
    "whisper_compute_type": (("transcribe", "whisper_compute_type"),),
    "whisper_batch_size": (("transcribe", "whisper_batch_size"),),
    "whisper_diarize": (("transcribe", "whisper_diarize"),),
    "whisper_hf_token": (("transcribe", "whisper_hf_token"),),
    "prompt": (("translate", "prompt"),),
    "target_lang": (("translate", "target_lang"),),
    "source_lang": (("translate", "source_lang"),),
    "model": (("translate", "vertex_model"),),
    "vertex_reasoning_effort": (("translate", "vertex_reasoning_effort"),),
    "llm_provider": (("translate", "llm_provider"),),
    "bilingual": (
        ("translate", "bilingual"),
        ("postprocess", "bilingual"),
    ),
    "keyframes": (("format", "keyframes"),),
    "start": (("transcribe", "start"),),
    "end": (("transcribe", "end"),),
    "chunk_size": (("translate", "chunk_size"),),
}


def load_cli_config(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    normalized: dict[str, dict[str, Any]] = {}
    supported_sections = ", ".join(sorted(COMMAND_OPTION_SPECS))
    for section_name, section_value in raw_config.items():
        if section_name not in COMMAND_OPTION_SPECS:
            raise CLIConfigError(
                f"unknown section [{section_name}]; supported sections: {supported_sections}"
            )
        if not isinstance(section_value, dict):
            raise CLIConfigError(f"section [{section_name}] must be a TOML table")
        normalized[section_name] = _normalize_section(section_name, section_value)
    return normalized


def _normalize_section(
    section_name: str, section_value: dict[str, Any]
) -> dict[str, Any]:
    specs = COMMAND_OPTION_SPECS[section_name]
    supported_aliases = {
        alias for option_spec in specs.values() for alias in option_spec.aliases
    }
    unknown_keys = sorted(set(section_value) - supported_aliases)
    if unknown_keys:
        supported = ", ".join(sorted(supported_aliases))
        unknown = ", ".join(unknown_keys)
        raise CLIConfigError(
            f"unknown key(s) in [{section_name}]: {unknown}; supported keys: {supported}"
        )

    normalized: dict[str, Any] = {}
    for parameter_name, option_spec in specs.items():
        present_aliases = [
            alias for alias in option_spec.aliases if alias in section_value
        ]
        if not present_aliases:
            continue
        if len(present_aliases) > 1:
            alias_list = ", ".join(present_aliases)
            raise CLIConfigError(
                f"section [{section_name}] defines the same option more than once: {alias_list}"
            )
        selected_alias = present_aliases[0]
        try:
            normalized[parameter_name] = option_spec.converter(
                section_value[selected_alias]
            )
        except CLIConfigError as exc:
            raise CLIConfigError(f"[{section_name}].{selected_alias}: {exc}") from exc
    return normalized


def apply_command_config(
    ctx: Any, command_name: str, values: dict[str, Any]
) -> dict[str, Any]:
    cli_config = (ctx.obj or {}).get("cli_config", {})
    section = cli_config.get(command_name, {})
    resolved = dict(values)

    for parameter_name, value in values.items():
        if parameter_name not in section:
            continue
        if ctx.get_parameter_source(parameter_name) == ParameterSource.DEFAULT:
            resolved[parameter_name] = section[parameter_name]
        else:
            resolved[parameter_name] = value
    return resolved


def apply_run_config(ctx: Any, values: dict[str, Any]) -> dict[str, Any]:
    cli_config = (ctx.obj or {}).get("cli_config", {})
    resolved = dict(values)

    for parameter_name, value in values.items():
        if ctx.get_parameter_source(parameter_name) != ParameterSource.DEFAULT:
            resolved[parameter_name] = value
            continue

        if parameter_name in cli_config.get("run", {}):
            resolved[parameter_name] = cli_config["run"][parameter_name]
            continue

        for section_name, section_key in RUN_SECTION_FALLBACKS.get(parameter_name, ()):
            section = cli_config.get(section_name, {})
            if section_key in section:
                resolved[parameter_name] = section[section_key]
                break

    return resolved
