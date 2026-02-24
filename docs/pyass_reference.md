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

## ASS Override Tags
The `pyass` library provides object-oriented implementations for most standard ASS override tags via subclasses of `pyass.tag.Tag`. Alternatively, you can always include raw tag strings (e.g., `{\an8\b1}`) directly within the `text` field of an `Event`.

Below is a quick reference of common ASS override tags and their effects:

### Styling
*   `\i1` / `\i0`: Italic on/off. (e.g., `ItalicTag`)
*   `\b1` / `\b0` / `\b<weight>`: Bold on/off, or specific weight (100-900). (e.g., `BoldTag`)
*   `\u1` / `\u0`: Underline on/off. (e.g., `UnderlineTag`)
*   `\s1` / `\s0`: Strikeout on/off. (e.g., `StrikeoutTag`)
*   `\fn<name>`: Font name (e.g., `\fnArial`). (e.g., `FontNameTag`)
*   `\fs<size>`: Font size. (e.g., `FontSizeTag`)
*   `\fscx<scale>` / `\fscy<scale>`: Font scale X/Y (percentage, e.g., 100 is normal). (e.g., `TextScaleTag`)
*   `\fsp<spacing>`: Letter spacing. (e.g., `TextSpacingTag`)

### Colors and Alpha
Color codes are in hexadecimal Blue-Green-Red order: `&H<bb><gg><rr>&`.
*   `\c` or `\1c`: Primary fill color. (e.g., `ColorTag`)
*   `\2c`: Secondary fill color (karaoke).
*   `\3c`: Border color.
*   `\4c`: Shadow color.
*   `\alpha&H<aa>&`: Set alpha for all components (00 is opaque, FF is transparent). (e.g., `AlphaTag`)
*   `\1a`, `\2a`, `\3a`, `\4a`: Set alpha for primary, secondary, border, and shadow.

### Borders, Shadows, and Blur
*   `\bord<size>`: Border width. (e.g., `BorderSizeTag`)
*   `\xbord<size>` / `\ybord<size>`: X/Y specific border width.
*   `\shad<depth>`: Shadow distance. (e.g., `ShadowDepthTag`)
*   `\xshad<depth>` / `\yshad<depth>`: X/Y specific shadow distance.
*   `\be<strength>`: Blur edges. (e.g., `BlurEdgesTag`)
*   `\blur<strength>`: Gaussian blur.

### Alignment and Positioning
*   `\an<pos>`: Numpad-style alignment (1-9). E.g., `\an2` is Bottom-Center. (e.g., `AlignmentTag`)
*   `\pos(<X>,<Y>)`: Set absolute position. (e.g., `PositionTag`)
*   `\move(<x1>,<y1>,<x2>,<y2>[,<t1>,<t2>])`: Move from (x1,y1) to (x2,y2). Optional t1/t2 specify time in ms. (e.g., `MoveTag`)
*   `\org(<X>,<Y>)`: Set rotation origin.

### Rotation and Transform
*   `\frz<amount>`, `\frx`, `\fry`: Rotate around Z, X, or Y axis in degrees. (e.g., `TextRotationTag`, `TextShearTag`)
*   `\t([<t1>,<t2>,[<accel>,]]<style modifiers>)`: Animated transform. Modifies style tags over time. (e.g., `TransformTag`)

### Timing and Effects
*   `\fad(<fadein>,<fadeout>)`: Simple fade in/out in milliseconds. (e.g., `FadeTag`)
*   `\fade(<a1>,<a2>,<a3>,<t1>,<t2>,<t3>,<t4>)`: Complex fade with alpha values and times. (e.g., `ComplexFadeTag`)
*   `\k<duration>`, `\K`, `\kf`: Karaoke effects. Duration is in centiseconds. (e.g., `KaraokeTag`)

### Clipping
*   `\clip(<x1>,<y1>,<x2>,<y2>)`: Rectangle clip (keeps inside). (e.g., `RectangularClipTag`)
*   `\iclip(<x1>,<y1>,<x2>,<y2>)`: Inverse rectangle clip (keeps outside).
*   `\clip([<scale>,]<drawing commands>)`: Vector drawing clip. (e.g., `DrawingClipTag`)
