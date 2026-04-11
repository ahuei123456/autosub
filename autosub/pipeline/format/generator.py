from pathlib import Path
from typing import List
import pyass

from autosub.core.schemas import SubtitleLine


def generate_ass_file(lines: List[SubtitleLine], output_path: Path):
    """
    Converts a list of SubtitleLine objects into a pyass Script and saves it to disk.
    Automatically generates unique styles per speaker.
    """
    # 1. Identify unique speakers and generate styles
    unique_speakers = {line.speaker if line.speaker else "Default" for line in lines}

    # Pre-defined array of subtle color tints for up to a few speakers
    speaker_colors = [
        pyass.Color(r=255, g=255, b=255, a=0),  # White
        pyass.Color(r=255, g=255, b=200, a=0),  # Light Yellow
        pyass.Color(r=200, g=255, b=255, a=0),  # Light Cyan
        pyass.Color(r=255, g=200, b=255, a=0),  # Light Magenta
        pyass.Color(r=200, g=255, b=200, a=0),  # Light Green
    ]

    styles = []
    speakerOriginToStyleMap = {}

    for i, speaker_name in enumerate(sorted(unique_speakers)):
        c = speaker_colors[i % len(speaker_colors)]
        style_name = speaker_name if speaker_name else "Default"

        # Build style
        st = pyass.Style(
            name=style_name,
            fontName="Arial",
            fontSize=48,
            isBold=True,
            primaryColor=c,
            outlineColor=pyass.Color(r=0, g=0, b=0, a=0),
            backColor=pyass.Color(r=0, g=0, b=0, a=0),
            outline=2.0,
            shadow=2.0,
            alignment=pyass.Alignment.BOTTOM,
            marginV=20,
        )
        styles.append(st)
        speakerOriginToStyleMap[speaker_name] = style_name

    # 2. Convert SubtitleLines into pyass Events
    pyass_events: List[pyass.Event] = []

    for line in lines:
        assigned_style = speakerOriginToStyleMap.get(
            line.speaker if line.speaker else "Default", "Default"
        )
        event_name = line.role or (line.speaker if line.speaker else "")

        if line.corner:
            pyass_events.append(
                pyass.Event(
                    format=pyass.EventFormat.COMMENT,
                    start=pyass.timedelta(seconds=line.start_time),
                    end=pyass.timedelta(seconds=line.end_time),
                    style=assigned_style,
                    effect="corner",
                    text=f"=== Corner: {line.corner} ===",
                )
            )

        pyass_events.append(
            pyass.Event(
                start=pyass.timedelta(seconds=line.start_time),
                end=pyass.timedelta(seconds=line.end_time),
                style=assigned_style,
                name=event_name,
                text=line.text,
            )
        )

    # 3. Create the pyass Script container
    script = pyass.Script(styles=styles, events=pyass_events)

    # 4. Dump to disk
    with open(output_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)
