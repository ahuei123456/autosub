import pyass
import pytest

from autosub.core.schemas import (
    SubtitleCue,
    SubtitleDocument,
    SubtitleLine,
)
from autosub.core.speaker_map import (
    build_speaker_prompt,
    hex_to_pyass_color,
    load_speaker_map,
    remap_speaker_labels,
)
from autosub.pipeline.format import generator


# --- Loading ---


def test_load_speaker_map(tmp_path):
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text(
        """\
[speakers."0"]
name = "Suzuki Minori"
character = "Ena Shinonome"
color = "#FFA0A0"

[speakers."1"]
name = "Sato Hinata"
character = "Mizuki Akiyama"
color = "#A0D0FF"
""",
        encoding="utf-8",
    )

    result = load_speaker_map(map_file)
    assert result["0"] == {
        "name": "Suzuki Minori",
        "character": "Ena Shinonome",
        "color": "#FFA0A0",
    }
    assert result["1"]["name"] == "Sato Hinata"


def test_load_speaker_map_optional_fields_default_to_none(tmp_path):
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text('[speakers."1"]\nname = "Speaker One"\n', encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result["1"]["color"] is None
    assert result["1"]["character"] is None


def test_load_speaker_map_falls_back_to_the_label_as_name(tmp_path):
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_text('[speakers."3"]\ncolor = "#00FF00"\n', encoding="utf-8")

    assert load_speaker_map(map_file)["3"]["name"] == "3"


def test_load_speaker_map_accepts_a_utf8_bom(tmp_path):
    """Aegisub and Windows editors routinely write one."""
    map_file = tmp_path / "speaker_map.toml"
    map_file.write_bytes(b'\xef\xbb\xbf[speakers."0"]\nname = "Kanade"\n')

    assert load_speaker_map(map_file)["0"]["name"] == "Kanade"


# --- Remapping ---


def test_remap_speaker_labels():
    lines = [
        SubtitleLine(text="hello", start_time=0.0, end_time=1.0, speaker="1"),
        SubtitleLine(text="world", start_time=1.0, end_time=2.0, speaker="2"),
        SubtitleLine(text="unlabelled", start_time=2.0, end_time=3.0, speaker=None),
        SubtitleLine(text="unmapped", start_time=3.0, end_time=4.0, speaker="9"),
    ]
    speaker_map = {"1": {"name": "Mizuki"}, "2": {"name": "Ena"}}

    remap_speaker_labels(lines, speaker_map)

    assert [line.speaker for line in lines] == ["Mizuki", "Ena", None, "9"]


def test_remap_speaker_labels_is_idempotent():
    """format may remap before building the document and again before rendering."""
    lines = [SubtitleLine(text="hi", start_time=0.0, end_time=1.0, speaker="1")]
    speaker_map = {"1": {"name": "Mizuki"}}

    remap_speaker_labels(lines, speaker_map)
    remap_speaker_labels(lines, speaker_map)

    assert lines[0].speaker == "Mizuki"


# --- Colors ---


@pytest.mark.parametrize(
    "value,expected",
    [("#FF8040", (255, 128, 64)), ("00FF00", (0, 255, 0))],
)
def test_hex_to_pyass_color(value, expected):
    c = hex_to_pyass_color(value)
    assert (c.r, c.g, c.b) == expected
    assert c.a == 0


# --- Prompt fragment ---


def test_build_speaker_prompt_with_characters():
    result = build_speaker_prompt(
        {
            "0": {"name": "Suzuki Minori", "character": "Ena Shinonome"},
            "1": {"name": "Sato Hinata", "character": "Mizuki Akiyama"},
        }
    )
    assert result.startswith("Speakers in this recording:")
    assert "- Suzuki Minori (voice of Ena Shinonome)" in result
    assert "- Sato Hinata (voice of Mizuki Akiyama)" in result


def test_build_speaker_prompt_without_characters():
    result = build_speaker_prompt({"0": {"name": "Speaker A", "character": None}})
    assert "- Speaker A" in result
    assert "voice of" not in result


# --- Styling ---


def _document(*speakers):
    return SubtitleDocument(
        stage="formatted",
        cues=[
            SubtitleCue(
                id=f"cue-{i + 1:05d}",
                start_time=float(i),
                end_time=float(i + 1),
                source_text=f"line {i}",
                speaker=s,
            )
            for i, s in enumerate(speakers)
        ],
    )


def _styles(path):
    return {
        line.split(",")[0].removeprefix("Style: ").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Style:")
    }


def test_render_names_styles_after_mapped_speakers(tmp_path):
    out = tmp_path / "out.ass"
    generator.render_ass_document(
        _document("0", "1"),
        out,
        mode="source",
        speaker_map={
            "0": {"name": "Kanade", "color": "#FFA0A0"},
            "1": {"name": "Mafuyu", "color": "#A0D0FF"},
        },
    )

    assert _styles(out) == {"Kanade", "Mafuyu"}


def test_render_uses_the_configured_color(tmp_path):
    out = tmp_path / "out.ass"
    generator.render_ass_document(
        _document("0"), out, mode="source", speaker_map={"0": {"name": "Kanade", "color": "#FF8040"}}
    )

    expected = pyass.Color(r=255, g=128, b=64, a=0)
    style = next(
        line
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.startswith("Style: Kanade")
    )
    assert str(expected) in style


def test_render_without_a_speaker_map_keeps_raw_labels(tmp_path):
    out = tmp_path / "out.ass"
    generator.render_ass_document(_document("0", "1"), out, mode="source")

    assert _styles(out) == {"0", "1"}


def test_a_speaker_without_a_color_still_gets_a_style(tmp_path):
    """Color is optional; the generated palette covers the gap."""
    out = tmp_path / "out.ass"
    generator.render_ass_document(
        _document("0"), out, mode="source", speaker_map={"0": {"name": "Kanade"}}
    )

    assert _styles(out) == {"Kanade"}


def test_many_labels_can_map_to_one_speaker(tmp_path):
    """Diarization over-segments; two labels for one person share a style."""
    out = tmp_path / "out.ass"
    generator.render_ass_document(
        _document("0", "1"),
        out,
        mode="source",
        speaker_map={
            "0": {"name": "Kanade", "color": "#FFA0A0"},
            "1": {"name": "Kanade", "color": "#FFA0A0"},
        },
    )

    assert _styles(out) == {"Kanade"}
    dialogue = [
        line
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.startswith("Dialogue:")
    ]
    assert len(dialogue) == 2
    assert all(",Kanade," in line for line in dialogue)


def test_generate_ass_file_applies_the_speaker_map(tmp_path):
    out = tmp_path / "out.ass"
    generator.generate_ass_file(
        [SubtitleLine(text="hi", start_time=0.0, end_time=1.0, speaker="0")],
        out,
        speaker_map={"0": {"name": "Kanade", "color": "#FFA0A0"}},
    )

    assert _styles(out) == {"Kanade"}
