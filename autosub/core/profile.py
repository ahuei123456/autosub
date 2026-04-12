import copy
import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_STAGES = ("transcribe", "format", "translate", "postprocess")


def _profile_search_dirs() -> tuple[Path, ...]:
    profiles_root = Path("profiles")
    return (profiles_root / "local", profiles_root / "examples", profiles_root)


def _resolve_profile_path(profile_name: str) -> Path | None:
    profile_filename = f"{profile_name}.toml"
    for directory in _profile_search_dirs():
        candidate = directory / profile_filename
        if candidate.exists():
            return candidate
    return None


def _resolve_prompt_path(prompt_value: str) -> Path | None:
    prompt_path = Path(prompt_value)
    if prompt_path.exists():
        return prompt_path

    if prompt_path.parts and prompt_path.parts[0] == "prompts":
        prompts_root = Path("prompts")
        relative_prompt_path = Path(*prompt_path.parts[1:])
        search_dirs = (
            prompts_root / "local",
            prompts_root / "examples",
            prompts_root,
        )
        for directory in search_dirs:
            candidate = directory / relative_prompt_path
            if candidate.exists():
                return candidate

    return None


def _merge_nested_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _empty_stage_profile() -> dict[str, dict]:
    return {
        "transcribe": {"vocab": []},
        "format": {"extensions": {}, "replacements": {}},
        "translate": {"prompt": [], "glossary": {}},
        "postprocess": {"extensions": {}},
    }


def _merge_stage_section(
    base: dict, override: dict, *, append_list_keys: tuple[str, ...] = ()
) -> dict:
    merged = _merge_nested_dict(base, override)
    for key in append_list_keys:
        merged[key] = [*base.get(key, []), *override.get(key, [])]
    return merged


def _merge_format_stage(base: dict, override: dict) -> dict:
    """Merge format stage with special handling for corners segment accumulation."""
    merged = _merge_stage_section(base, override)

    # Accumulate corners segments from base and override (don't replace)
    base_segments = (
        base.get("extensions", {}).get("corners", {}).get("segments", [])
    )
    override_segments = (
        override.get("extensions", {}).get("corners", {}).get("segments", [])
    )
    if base_segments or override_segments:
        merged.setdefault("extensions", {}).setdefault("corners", {})["segments"] = [
            *base_segments,
            *override_segments,
        ]

    return merged


def _merge_profiles(
    base: dict[str, dict], override: dict[str, dict]
) -> dict[str, dict]:
    return {
        "transcribe": _merge_stage_section(
            base["transcribe"], override["transcribe"], append_list_keys=("vocab",)
        ),
        "format": _merge_format_stage(base["format"], override["format"]),
        "translate": _merge_stage_section(
            base["translate"], override["translate"], append_list_keys=("prompt",)
        ),
        "postprocess": _merge_stage_section(
            base["postprocess"], override["postprocess"]
        ),
    }


def _load_prompt_fragments(
    profile_name: str, value: object, *, key_name: str
) -> list[str]:
    if isinstance(value, str):
        raw_parts = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        raw_parts = value
    else:
        logger.warning(
            f"'{key_name}' in {profile_name} must be a string or list of strings."
        )
        return []

    prompt_parts: list[str] = []
    for raw_part in raw_parts:
        prompt_value = raw_part.strip()
        if prompt_value.endswith(".md") or prompt_value.endswith(".txt"):
            prompt_file_path = _resolve_prompt_path(prompt_value)
            if prompt_file_path is not None:
                with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
                    prompt_parts.append(prompt_file.read().strip())
            else:
                logger.warning(
                    f"Prompt file {prompt_value} referenced by {profile_name} not found."
                )
        else:
            prompt_parts.append(prompt_value)
    return prompt_parts


def _copy_unstructured_stage_keys(
    section_value: dict, *, excluded_keys: set[str]
) -> dict[str, object]:
    return {
        key: copy.deepcopy(value)
        for key, value in section_value.items()
        if key not in excluded_keys
    }


def _normalize_stage_section(
    profile_name: str, stage_name: str, section_value: object
) -> dict[str, object]:
    if not isinstance(section_value, dict):
        logger.warning(f"'{stage_name}' in {profile_name} must be a TOML table/dict.")
        return {}

    if stage_name == "transcribe":
        normalized = _copy_unstructured_stage_keys(
            section_value, excluded_keys={"vocab"}
        )
        if "vocab" in section_value:
            if isinstance(section_value["vocab"], list):
                normalized["vocab"] = [str(item) for item in section_value["vocab"]]
            else:
                logger.warning(
                    f"'transcribe.vocab' in {profile_name} must be a list of strings."
                )
        return normalized

    if stage_name == "format":
        normalized = _copy_unstructured_stage_keys(
            section_value, excluded_keys={"extensions", "replacements"}
        )
        if "extensions" in section_value:
            if isinstance(section_value["extensions"], dict):
                normalized["extensions"] = copy.deepcopy(section_value["extensions"])
            else:
                logger.warning(
                    f"'format.extensions' in {profile_name} must be a TOML table/dict."
                )
        if "replacements" in section_value:
            if isinstance(section_value["replacements"], dict):
                normalized["replacements"] = copy.deepcopy(
                    section_value["replacements"]
                )
            else:
                logger.warning(
                    f"'format.replacements' in {profile_name} must be a TOML table/dict."
                )
        return normalized

    if stage_name == "translate":
        normalized = _copy_unstructured_stage_keys(
            section_value, excluded_keys={"prompt", "glossary"}
        )
        if "prompt" in section_value:
            normalized["prompt"] = _load_prompt_fragments(
                profile_name, section_value["prompt"], key_name="translate.prompt"
            )
        if "glossary" in section_value:
            if isinstance(section_value["glossary"], dict):
                normalized["glossary"] = copy.deepcopy(section_value["glossary"])
            else:
                logger.warning(
                    f"'translate.glossary' in {profile_name} must be a TOML table/dict."
                )
        return normalized

    normalized = _copy_unstructured_stage_keys(
        section_value, excluded_keys={"extensions"}
    )
    if "extensions" in section_value:
        if isinstance(section_value["extensions"], dict):
            normalized["extensions"] = copy.deepcopy(section_value["extensions"])
        else:
            logger.warning(
                f"'postprocess.extensions' in {profile_name} must be a TOML table/dict."
            )
    return normalized


def _normalize_profile_data(profile_name: str, data: dict) -> dict[str, dict]:
    normalized = _empty_stage_profile()

    for stage_name in PROFILE_STAGES:
        if stage_name not in data:
            continue
        normalized[stage_name] = _merge_stage_section(
            normalized[stage_name],
            _normalize_stage_section(profile_name, stage_name, data[stage_name]),
            append_list_keys=("vocab",)
            if stage_name == "transcribe"
            else ("prompt",)
            if stage_name == "translate"
            else (),
        )

    if "prompt" in data:
        normalized["translate"]["prompt"] = [
            *normalized["translate"].get("prompt", []),
            *_load_prompt_fragments(profile_name, data["prompt"], key_name="prompt"),
        ]

    if "vocab" in data:
        if isinstance(data["vocab"], list):
            normalized["transcribe"]["vocab"] = [
                *normalized["transcribe"].get("vocab", []),
                *(str(item) for item in data["vocab"]),
            ]
        else:
            logger.warning(f"'vocab' in {profile_name} must be a list of strings.")

    if "timing" in data:
        if isinstance(data["timing"], dict):
            normalized["format"] = _merge_nested_dict(
                normalized["format"], copy.deepcopy(data["timing"])
            )
        else:
            logger.warning(f"'timing' in {profile_name} must be a TOML table/dict.")

    if "extensions" in data:
        if isinstance(data["extensions"], dict):
            legacy_extensions = copy.deepcopy(data["extensions"])
            normalized["format"]["extensions"] = _merge_nested_dict(
                normalized["format"].get("extensions", {}), legacy_extensions
            )
            normalized["postprocess"]["extensions"] = _merge_nested_dict(
                normalized["postprocess"].get("extensions", {}), legacy_extensions
            )
        else:
            logger.warning(f"'extensions' in {profile_name} must be a TOML table/dict.")

    if "glossary" in data:
        if isinstance(data["glossary"], dict):
            normalized["translate"]["glossary"] = _merge_nested_dict(
                normalized["translate"].get("glossary", {}),
                copy.deepcopy(data["glossary"]),
            )
        else:
            logger.warning(f"'glossary' in {profile_name} must be a TOML table/dict.")

    if "replacements" in data:
        if isinstance(data["replacements"], dict):
            normalized["format"]["replacements"] = _merge_nested_dict(
                normalized["format"].get("replacements", {}),
                copy.deepcopy(data["replacements"]),
            )
        else:
            logger.warning(
                f"'replacements' in {profile_name} must be a TOML table/dict."
            )

    if "corners" in data:
        if isinstance(data["corners"], list):
            # Auto-convert top-level [[corners]] to format extension config
            corners_ext = normalized["format"].setdefault("extensions", {}).setdefault(
                "corners", {}
            )
            corners_ext.setdefault("enabled", True)
            corners_ext.setdefault("engine", "hybrid")
            existing_segments = corners_ext.get("segments", [])
            corners_ext["segments"] = existing_segments + data["corners"]
        else:
            logger.warning(f"'corners' in {profile_name} must be an array of tables.")

    return normalized


def _load_profile_sections(
    profile_name: str, visited: set[str] | None = None
) -> dict[str, dict]:
    if visited is None:
        visited = set()

    if profile_name in visited:
        return _empty_stage_profile()
    visited.add(profile_name)

    profile_path = _resolve_profile_path(profile_name)
    if profile_path is None:
        logger.warning(
            f"Profile {profile_name}.toml not found in profiles/local, "
            "profiles/examples, or profiles/."
        )
        return _empty_stage_profile()

    try:
        with open(profile_path, "rb") as handle:
            data = tomllib.load(handle)
    except Exception as exc:
        logger.error(f"Failed to parse TOML profile {profile_path}: {exc}")
        return _empty_stage_profile()

    combined = _empty_stage_profile()
    for base_profile in data.get("extends", []):
        base_data = _load_profile_sections(base_profile, visited)
        combined = _merge_profiles(combined, base_data)

    merged = _merge_profiles(combined, _normalize_profile_data(profile_name, data))

    return merged


def load_unified_profile(profile_name: str, visited: set[str] | None = None) -> dict:
    """
    Loads a TOML profile recursively, resolving `extends` arrays.
    Normalizes both staged and legacy flat profile keys into stage-specific sections.
    """
    staged = _load_profile_sections(profile_name, visited)
    format_stage = staged["format"]
    translate_stage = staged["translate"]
    postprocess_stage = staged["postprocess"]
    legacy_timing = {
        key: value
        for key, value in format_stage.items()
        if key not in {"extensions", "replacements"}
    }
    legacy_extensions = _merge_nested_dict(
        format_stage.get("extensions", {}),
        postprocess_stage.get("extensions", {}),
    )
    return {
        **staged,
        "prompt": list(translate_stage.get("prompt", [])),
        "vocab": list(staged["transcribe"].get("vocab", [])),
        "timing": legacy_timing,
        "extensions": legacy_extensions,
        "glossary": copy.deepcopy(translate_stage.get("glossary", {})),
        "replacements": copy.deepcopy(format_stage.get("replacements", {})),
        "corners": format_stage.get("extensions", {}).get("corners", {}).get("segments", []),
    }
