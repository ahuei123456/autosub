# pyass Library Reference

This is an internally generated technical reference for the `xicearcher/pyass` library, created by analyzing its source code. It documents the core data models and functions required to programmatically generate Advanced SubStation Alpha (`.ass`) files.

## Overview
The library revolves around the `Script` class, which acts as the root container for `.ass` file sections (`ScriptInfo`, `Styles`, `Events`).

### Core Modules
```python
import pyass
from pyass import Script, Style, Event
from pyass import timedelta
from pyass import Color
```

## `Script` (Root Document)
The `Script` class represents a full `.ass` file.
*   **Initialization**: `Script(styles=[...], events=[...])`
    *   It automatically populates `ScriptInfo` with default AEG/ASS boilerplate (e.g. `Script Type: v4.00+`).
*   **Properties**:
    *   `styles`: A list of `Style` objects.
    *   `events`: A list of `Event` objects.
*   **Output functions**:
    *   `pyass.dump(script, file_object)`
    *   `pyass.dumps(script) -> str`

## `Style` (Subtitle Styles)
Defines the visual appearance of a class of subtitles.
*   **Initialization**: `Style(name="MyStyle", fontName="Arial", fontSize=48, ...)`
*   **Key Parameters**:
    *   `name` (str): Style name (default: "Default")
    *   `fontName` (str): Font family (default: "Arial")
    *   `fontSize` (int): Font size (default: 48)
    *   `primaryColor` (Color): Primary text color
    *   `outlineColor` (Color): Border color
    *   `backColor` (Color): Shadow color
    *   `isBold`, `isItalic` (bool)
    *   `outline`, `shadow` (float): Border and shadow thickness
    *   `alignment` (pyass.Alignment or int): NumPad text alignment (e.g., 2 is Bottom-Center)
    *   `marginL`, `marginR`, `marginV` (int): Margins

## `Event` (Subtitle Lines)
Represents a single line/event in the subtitle file.
*   **Initialization**: `Event(start=timedelta(seconds=1), end=timedelta(seconds=5), text="Hello World", ...)`
*   **Key Parameters**:
    *   `format` (EventFormat): Default is `EventFormat.DIALOGUE`.
    *   `start`, `end` (`pyass.timedelta`): Start and end times. Note that `pyass.timedelta` inherits directly from `datetime.timedelta`.
    *   `style` (str): Matches the `name` of a defined `Style` (default: "Default").
    *   `name` (str): Speaker name.
    *   `text` (str): The raw text of the subtitle, which can include formatting `{tags}`.

## `timedelta` (Time handling)
`pyass` extends `datetime.timedelta` with `pyass.timedelta`.
*   Can be instantiated identical to standard timedelta: `pyass.timedelta(hours=1, minutes=2, seconds=3, milliseconds=400)`
*   Also supports formatting via `str(td)` to produce the required ASS timestamp `H:MM:SS.cs`.

## Example Usage
```python
import pyass

# 1. Define a global style
my_style = pyass.Style(
    name="SpeakerOne",
    fontName="Roboto",
    fontSize=65,
    isBold=True,
    primaryColor=pyass.Color(r=255, g=255, b=255)
)

# 2. Define events (lines)
events = [
    pyass.Event(
        start=pyass.timedelta(seconds=1.5),
        end=pyass.timedelta(seconds=4.2),
        style="SpeakerOne",
        text="Hello, this is my first line!"
    )
]

# 3. Create script and dump to file
script = pyass.Script(styles=[my_style], events=events)

with open("output.ass", "w", encoding="utf-8") as f:
    pyass.dump(script, f)
```
