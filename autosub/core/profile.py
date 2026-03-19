import tomllib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_unified_profile(profile_name: str, visited: set[str] | None = None) -> dict:
    """
    Loads a TOML profile recursively, resolving 'extends' arrays.
    Combines 'prompt' into a list of strings and 'vocab' into a single list.
    If a 'prompt' ends with .md or .txt, it loads the file contents.
    """
    if visited is None:
        visited = set()

    if profile_name in visited:
        return {"prompt": [], "vocab": [], "speakers": None, "timing": {}}
    visited.add(profile_name)

    profile_path = Path("profiles") / f"{profile_name}.toml"
    if not profile_path.exists():
        logger.warning(f"Profile {profile_name}.toml not found in profiles/ directory.")
        return {"prompt": [], "vocab": [], "speakers": None, "timing": {}}

    try:
        with open(profile_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.error(f"Failed to parse TOML profile {profile_path}: {e}")
        return {"prompt": [], "vocab": [], "speakers": None, "timing": {}}

    combined_prompt = []
    combined_vocab = []
    final_speakers = None
    final_timing = {}

    # 1. Process base profiles recursively (so base instructions come first)
    for base in data.get("extends", []):
        base_data = load_unified_profile(base, visited)
        combined_prompt.extend(base_data["prompt"])
        combined_vocab.extend(base_data["vocab"])
        if final_speakers is None and base_data.get("speakers") is not None:
            final_speakers = base_data["speakers"]
        # Update inherited timing. Top level overrides base.
        for k, v in base_data.get("timing", {}).items():
            if k not in final_timing:
                final_timing[k] = v

    # 2. Append this profile's data
    if "prompt" in data:
        p_val = data["prompt"].strip()
        if p_val.endswith(".md") or p_val.endswith(".txt"):
            prompt_file_path = Path(p_val)
            if prompt_file_path.exists():
                with open(prompt_file_path, "r", encoding="utf-8") as pf:
                    combined_prompt.append(pf.read().strip())
            else:
                logger.warning(
                    f"Prompt file {prompt_file_path} referenced by {profile_name} not found."
                )
        else:
            combined_prompt.append(p_val)

    if "vocab" in data:
        if isinstance(data["vocab"], list):
            combined_vocab.extend(str(v) for v in data["vocab"])
        else:
            logger.warning(f"'vocab' in {profile_name} must be a list of strings.")

    # Override speakers if explicitly defined in this profile layer
    if "speakers" in data:
        if isinstance(data["speakers"], int):
            final_speakers = data["speakers"]
        else:
            logger.warning(f"'speakers' in {profile_name} must be an integer.")

    if "timing" in data:
        if isinstance(data["timing"], dict):
            final_timing.update(data["timing"])
        else:
            logger.warning(f"'timing' in {profile_name} must be a TOML table/dict.")

    return {
        "prompt": combined_prompt,
        "vocab": combined_vocab,
        "speakers": final_speakers,
        "timing": final_timing,
    }
