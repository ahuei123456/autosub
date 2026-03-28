import logging
from pathlib import Path

import pyass
import yaml

from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)


def load_speaker_map(path: Path) -> dict[str, dict]:
    """Load a speaker_map.yaml file.

    Expected format:
        speakers:
          "1":
            name: "Mizuki Akiyama"
            color: "#FFFF00"
          "2":
            name: "Ena Shinonome"
            color: "#FF8080"

    Returns {"1": {"name": "Mizuki Akiyama", "color": "#FFFF00"}, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    speakers = data.get("speakers", {})
    result = {}
    for label, entry in speakers.items():
        result[str(label)] = {
            "name": entry.get("name", str(label)),
            "color": entry.get("color"),
        }
    return result


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
