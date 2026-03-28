import pyass

from autosub.core.schemas import SubtitleLine, TranscribedWord
from autosub.core.speaker_map import hex_to_pyass_color, load_speaker_map, remap_speaker_labels
from autosub.pipeline.format.chunker import chunk_words_to_lines
from autosub.core.profile import load_unified_profile


def test_chunk_words_by_speaker():
    # Simulate overlapping speech from 2 speakers
    words = [
        TranscribedWord(
            word="Hello,", start_time=0.0, end_time=0.5, speaker="Speaker_1"
        ),
        TranscribedWord(word="Hi", start_time=0.2, end_time=0.4, speaker="Speaker_2"),
        TranscribedWord(word="how", start_time=0.5, end_time=0.8, speaker="Speaker_1"),
        TranscribedWord(
            word="there!", start_time=0.4, end_time=0.9, speaker="Speaker_2"
        ),
        TranscribedWord(word="are", start_time=0.8, end_time=1.0, speaker="Speaker_1"),
        TranscribedWord(word="you?", start_time=1.0, end_time=1.5, speaker="Speaker_1"),
    ]

    lines = chunk_words_to_lines(words)

    # We expect 2 lines, one for speaker 1 and one for speaker 2.
    assert len(lines) == 2

    l1 = lines[
        0
    ]  # Speaker 2 starts earliest logically but we sort chronologically by line start time. Speaker 1 starts at 0.0
    assert l1.speaker == "Speaker_1"
    assert l1.text == "Hello,howareyou?"

    l2 = lines[1]  # Speaker 2 starts at 0.2
    assert l2.speaker == "Speaker_2"
    assert l2.text == "Hithere!"


def test_soft_punctuation_splits_dense_chunk_without_pause():
    words = [
        TranscribedWord(word=f"語{i}、", start_time=i * 0.1, end_time=i * 0.1 + 0.05)
        for i in range(16)
    ]
    words.extend(
        [
            TranscribedWord(word="続き", start_time=1.6, end_time=1.65),
            TranscribedWord(word="です", start_time=1.7, end_time=1.75),
        ]
    )

    lines = chunk_words_to_lines(words)

    assert len(lines) == 2
    assert lines[0].text == "".join(word.word for word in words[:16])
    assert lines[1].text == "続きです"


def test_soft_punctuation_does_not_split_short_chunk_without_pause():
    words = [
        TranscribedWord(word=f"語{i}、", start_time=i * 0.1, end_time=i * 0.1 + 0.05)
        for i in range(11)
    ]
    words.append(TranscribedWord(word="続き", start_time=1.1, end_time=1.15))

    lines = chunk_words_to_lines(words)

    assert len(lines) == 1
    assert lines[0].text == "".join(word.word for word in words)


def test_profile_prompt_inheritance(tmp_path):
    profile_dir = tmp_path / "profiles"
    prompt_dir = tmp_path / "prompts"
    prompt_examples_dir = prompt_dir / "examples"
    profile_dir.mkdir()
    prompt_examples_dir.mkdir(parents=True)

    (prompt_examples_dir / "base.md").write_text("base guidance", encoding="utf-8")
    (prompt_examples_dir / "child.md").write_text("child guidance", encoding="utf-8")

    (profile_dir / "base.toml").write_text(
        '[translate]\nprompt = "prompts/base.md"\n', encoding="utf-8"
    )
    (profile_dir / "child.toml").write_text(
        'extends = ["base"]\n[translate]\nprompt = "prompts/child.md"\n',
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts":
                return prompt_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["translate"]["prompt"] == ["base guidance", "child guidance"]
        assert data["prompt"] == ["base guidance", "child guidance"]
    finally:
        autosub.core.profile.Path = original_path


def test_legacy_flat_profile_keys_are_mapped_to_stages(tmp_path):
    profile_dir = tmp_path / "profiles"
    prompt_dir = tmp_path / "prompts"
    prompt_examples_dir = prompt_dir / "examples"
    profile_dir.mkdir()
    prompt_examples_dir.mkdir(parents=True)

    (prompt_examples_dir / "legacy.md").write_text("legacy guidance", encoding="utf-8")
    (profile_dir / "legacy.toml").write_text(
        """
prompt = "prompts/legacy.md"
vocab = ["鈴原希実"]

[timing]
min_duration_ms = 900

[extensions.radio_discourse]
enabled = true

[glossary]
"鈴原希実" = "Suzuhara Nozomi"

[replacements]
"鈴原のぞみ" = "鈴原希実"
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts":
                return prompt_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("legacy")
        assert data["transcribe"]["vocab"] == ["鈴原希実"]
        assert data["translate"]["prompt"] == ["legacy guidance"]
        assert data["translate"]["glossary"] == {"鈴原希実": "Suzuhara Nozomi"}
        assert data["format"]["min_duration_ms"] == 900
        assert data["format"]["replacements"] == {"鈴原のぞみ": "鈴原希実"}
        assert data["normalizer"] == {}
        assert data["format"]["extensions"]["radio_discourse"]["enabled"] is True
        assert data["postprocess"]["extensions"]["radio_discourse"]["enabled"] is True
    finally:
        autosub.core.profile.Path = original_path


def test_local_prompt_overrides_example_prompt(tmp_path):
    profiles_root = tmp_path / "profiles"
    profile_examples_dir = profiles_root / "examples"
    prompts_root = tmp_path / "prompts"
    prompt_local_dir = prompts_root / "local"
    prompt_examples_dir = prompts_root / "examples"
    profile_examples_dir.mkdir(parents=True)
    prompt_local_dir.mkdir(parents=True)
    prompt_examples_dir.mkdir(parents=True)

    (profile_examples_dir / "child.toml").write_text(
        '[translate]\nprompt = "prompts/child.md"\n', encoding="utf-8"
    )
    (prompt_examples_dir / "child.md").write_text("example guidance", encoding="utf-8")
    (prompt_local_dir / "child.md").write_text("local guidance", encoding="utf-8")

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profiles_root
            if args and args[0] == "prompts":
                return prompts_root
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["translate"]["prompt"] == ["local guidance"]
        assert data["prompt"] == ["local guidance"]
    finally:
        autosub.core.profile.Path = original_path


# --- Speaker map tests ---


def test_load_speaker_map(tmp_path):
    yaml_content = """\
speakers:
  "1":
    name: "Mizuki Akiyama"
    color: "#FFFF00"
  "2":
    name: "Ena Shinonome"
    color: "#FF8080"
"""
    map_file = tmp_path / "speaker_map.yaml"
    map_file.write_text(yaml_content, encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result == {
        "1": {"name": "Mizuki Akiyama", "color": "#FFFF00"},
        "2": {"name": "Ena Shinonome", "color": "#FF8080"},
    }


def test_load_speaker_map_missing_color(tmp_path):
    yaml_content = """\
speakers:
  "1":
    name: "Speaker One"
"""
    map_file = tmp_path / "speaker_map.yaml"
    map_file.write_text(yaml_content, encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result["1"]["name"] == "Speaker One"
    assert result["1"]["color"] is None


def test_load_speaker_map_fallback_name(tmp_path):
    yaml_content = """\
speakers:
  "3":
    color: "#00FF00"
"""
    map_file = tmp_path / "speaker_map.yaml"
    map_file.write_text(yaml_content, encoding="utf-8")

    result = load_speaker_map(map_file)
    assert result["3"]["name"] == "3"


def test_remap_speaker_labels():
    lines = [
        SubtitleLine(text="hello", start_time=0.0, end_time=1.0, speaker="1"),
        SubtitleLine(text="world", start_time=1.0, end_time=2.0, speaker="2"),
        SubtitleLine(text="test", start_time=2.0, end_time=3.0, speaker=None),
    ]
    speaker_map = {
        "1": {"name": "Mizuki", "color": "#FFFF00"},
        "2": {"name": "Ena", "color": "#FF8080"},
    }

    remap_speaker_labels(lines, speaker_map)

    assert lines[0].speaker == "Mizuki"
    assert lines[1].speaker == "Ena"
    assert lines[2].speaker is None  # unchanged


def test_hex_to_pyass_color():
    c = hex_to_pyass_color("#FF8040")
    assert c.r == 255
    assert c.g == 128
    assert c.b == 64
    assert c.a == 0


def test_hex_to_pyass_color_no_hash():
    c = hex_to_pyass_color("00FF00")
    assert c.r == 0
    assert c.g == 255
    assert c.b == 0
