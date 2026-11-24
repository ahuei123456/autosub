from pathlib import Path
from typing import Literal, NamedTuple
import pyass

from autosub.core.schemas import SubtitleCue, SubtitleDocument, SubtitleLine
from autosub.core.speaker_map import hex_to_pyass_color, remap_speaker_labels


AssRenderMode = Literal["source", "translated", "bilingual", "final"]


class _AssEntry(NamedTuple):
    text: str
    start_time: float
    end_time: float
    speaker: str | None = None
    role: str | None = None
    corner: str | None = None


def generate_ass_file(
    lines: list[SubtitleLine],
    output_path: Path,
    speaker_map: dict[str, dict] | None = None,
):
    """
    Converts a list of SubtitleLine objects into a pyass Script and saves it to disk.
    Automatically generates unique styles per speaker.

    If speaker_map is provided, styles are named after the mapped speakers and
    use their configured colors instead of raw labels with generated colors.
    """
    if speaker_map:
        remap_speaker_labels(lines, speaker_map)

    _write_script(
        _script_from_entries(
            [_line_to_entry(line) for line in lines], speaker_map=speaker_map
        ),
        output_path,
    )


def render_ass_document(
    document: SubtitleDocument,
    output_path: Path,
    *,
    mode: AssRenderMode,
    chunk_boundaries: list[int] | set[int] | None = None,
    speaker_map: dict[str, dict] | None = None,
) -> None:
    """Render a structured subtitle document into an ASS byproduct."""
    entries = [
        _AssEntry(
            text=_cue_text_for_mode(cue, mode),
            start_time=cue.start_time,
            end_time=cue.end_time,
            speaker=cue.speaker,
            role=cue.role,
            corner=cue.corner,
        )
        for cue in document.cues
    ]

    script = _script_from_entries(entries, speaker_map=speaker_map)
    boundaries = (
        document.chunk_boundaries if chunk_boundaries is None else chunk_boundaries
    )
    if boundaries:
        script.events = _insert_chunk_boundary_comments(script.events, set(boundaries))
    _write_script(script, output_path)


def _write_script(script: pyass.Script, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        pyass.dump(script, f)


def _cue_text_for_mode(cue: SubtitleCue, mode: AssRenderMode) -> str:
    source = cue.normalized_source_text or cue.source_text
    translated = cue.translated_text or source
    final = cue.final_text or translated

    if mode == "source":
        return source
    if mode == "translated":
        return translated
    if mode == "final":
        return final
    if mode == "bilingual":
        return rf"{{\fs24\a6}}{source}{{\N}}{{\fs48\a2}}{final}"
    raise ValueError(f"Unknown ASS render mode: {mode}")


def _line_to_entry(line: SubtitleLine) -> _AssEntry:
    return _AssEntry(
        text=line.text,
        start_time=line.start_time,
        end_time=line.end_time,
        speaker=line.speaker,
        role=line.role,
        corner=line.corner,
    )


def _script_from_entries(
    entries: list[_AssEntry],
    speaker_map: dict[str, dict] | None = None,
) -> pyass.Script:
    unique_speakers = {
        entry.speaker if entry.speaker else "Default" for entry in entries
    }
    speaker_colors = [
        pyass.Color(r=255, g=255, b=255, a=0),
        pyass.Color(r=255, g=255, b=200, a=0),
        pyass.Color(r=200, g=255, b=255, a=0),
        pyass.Color(r=255, g=200, b=255, a=0),
        pyass.Color(r=200, g=255, b=200, a=0),
    ]

    # A speaker may reach here as its raw diarization label or as the name it
    # was remapped to, depending on whether the caller remapped first, so both
    # are accepted as keys.
    mapped_colors: dict[str, pyass.Color] = {}
    label_to_name: dict[str, str] = {}
    for label, entry in (speaker_map or {}).items():
        name = entry.get("name", label)
        label_to_name[label] = name
        label_to_name[name] = name
        if entry.get("color"):
            color = hex_to_pyass_color(entry["color"])
            mapped_colors[label] = color
            mapped_colors[name] = color

    styles = []
    speaker_origin_to_style_map = {}
    for i, speaker_name in enumerate(sorted(unique_speakers)):
        resolved_name = label_to_name.get(speaker_name, speaker_name)
        c = mapped_colors.get(
            speaker_name,
            mapped_colors.get(resolved_name, speaker_colors[i % len(speaker_colors)]),
        )
        style_name = resolved_name if resolved_name else "Default"
        styles.append(
            pyass.Style(
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
        )
        speaker_origin_to_style_map[speaker_name] = style_name
        speaker_origin_to_style_map[resolved_name] = style_name

    pyass_events: list[pyass.Event] = []
    for entry in entries:
        raw_speaker = entry.speaker if entry.speaker else "Default"
        resolved_name = label_to_name.get(raw_speaker, raw_speaker)
        assigned_style = speaker_origin_to_style_map.get(
            resolved_name,
            speaker_origin_to_style_map.get(raw_speaker, "Default"),
        )
        event_name = entry.role or (resolved_name if resolved_name else "")

        if entry.corner:
            pyass_events.append(
                pyass.Event(
                    format=pyass.EventFormat.COMMENT,
                    start=pyass.timedelta(seconds=entry.start_time),
                    end=pyass.timedelta(seconds=entry.end_time),
                    style=assigned_style,
                    effect="corner",
                    text=f"=== Corner: {entry.corner} ===",
                )
            )

        pyass_events.append(
            pyass.Event(
                start=pyass.timedelta(seconds=entry.start_time),
                end=pyass.timedelta(seconds=entry.end_time),
                style=assigned_style,
                name=event_name,
                text=entry.text,
            )
        )

    return pyass.Script(styles=styles, events=pyass_events)


def _insert_chunk_boundary_comments(
    events: list[pyass.Event],
    chunk_boundaries: set[int],
) -> list[pyass.Event]:
    new_events: list[pyass.Event] = []
    dialogue_idx = 0
    for event in events:
        if isinstance(event, pyass.Event) and event.format != pyass.EventFormat.COMMENT:
            if dialogue_idx in chunk_boundaries:
                new_events.append(
                    pyass.Event(
                        format=pyass.EventFormat.COMMENT,
                        start=event.start,
                        end=event.end,
                        style=event.style,
                        effect="",
                        text="[autosub] Chunk boundary - review translation around this line",
                    )
                )
            dialogue_idx += 1
        new_events.append(event)
    return new_events
