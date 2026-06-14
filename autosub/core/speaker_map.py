import logging
import tomllib
from pathlib import Path

import pyass

from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)


def load_speaker_map(path: Path) -> dict[str, dict]:
    """Load a speaker_map.toml file.

    Expected format:
        [speakers."0"]
        name = "Suzuki Minori"
        character = "Ena Shinonome"
        color = "#FFA0A0"

        [speakers."1"]
        name = "Sato Hinata"
        character = "Mizuki Akiyama"
        color = "#A0D0FF"

    Returns {"0": {"name": ..., "character": ..., "color": ...}, ...}
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    speakers = data.get("speakers", {})
    result = {}
    for label, entry in speakers.items():
        result[str(label)] = {
            "name": entry.get("name", str(label)),
            "character": entry.get("character"),
            "color": entry.get("color"),
        }
    return result


def build_speaker_prompt(speaker_map: dict[str, dict]) -> str:
    """Build a prompt fragment describing the speakers in this recording."""
    lines = ["Speakers in this recording:"]
    for entry in speaker_map.values():
        name = entry["name"]
        character = entry.get("character")
        if character:
            lines.append(f"- {name} (voice of {character})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def remap_speaker_labels(lines: list[SubtitleLine], speaker_map: dict[str, dict]) -> None:
    """Replace raw API speaker labels with character names. Mutates in place."""
    for line in lines:
        if line.speaker and line.speaker in speaker_map:
            line.speaker = speaker_map[line.speaker]["name"]


def hex_to_pyass_color(hex_color: str) -> pyass.Color:
    """Convert '#RRGGBB' hex string to pyass.Color."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return pyass.Color(r=r, g=g, b=b, a=0)
