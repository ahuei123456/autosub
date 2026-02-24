from pathlib import Path
from typing import List
import pyass

from autosub.core.schemas import SubtitleLine


def generate_ass_file(lines: List[SubtitleLine], output_path: Path):
    """
    Converts a list of SubtitleLine objects into a pyass Script and saves it to disk.
    """
    # 1. Define a default style for the subtitles
    default_style = pyass.Style(
        name="Default",
        fontName="Arial",
        fontSize=48,
        isBold=True,
        primaryColor=pyass.Color(r=255, g=255, b=255, a=0),
        outlineColor=pyass.Color(r=0, g=0, b=0, a=0),
        backColor=pyass.Color(r=0, g=0, b=0, a=0),
        outline=2.0,
        shadow=2.0,
        alignment=pyass.Alignment.BOTTOM,
        marginV=20,
    )

    # 2. Convert SubtitleLines into pyass Events
    pyass_events: List[pyass.Event] = []

    for line in lines:
        pyass_events.append(
            pyass.Event(
                start=pyass.timedelta(seconds=line.start_time),
                end=pyass.timedelta(seconds=line.end_time),
                style="Default",
                name=line.speaker if line.speaker else "",
                text=line.text,
            )
        )

    # 3. Create the pyass Script container
    script = pyass.Script(styles=[default_style], events=pyass_events)

    # 4. Dump to disk
    with open(output_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)
