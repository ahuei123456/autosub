import pytest

from autosub.core.profile import load_unified_profile


def test_profile_extensions_are_inherited_and_overridden(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "base.toml").write_text(
        """
[format.extensions.radio_discourse]
enabled = true
split_framing_phrases = true
label_roles = false

[postprocess.extensions.radio_discourse]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    (profile_dir / "child.toml").write_text(
        """
extends = ["base"]

[format.extensions.radio_discourse]
label_roles = true
window_size = 8

[postprocess.extensions.radio_discourse]
preserve_quotes = true
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["format"]["extensions"]["radio_discourse"] == {
            "enabled": True,
            "split_framing_phrases": True,
            "label_roles": True,
            "window_size": 8,
        }
        assert data["postprocess"]["extensions"]["radio_discourse"] == {
            "enabled": True,
            "preserve_quotes": True,
        }
    finally:
        autosub.core.profile.Path = original_path


def test_local_profiles_can_extend_example_profiles(tmp_path):
    profiles_root = tmp_path / "profiles"
    local_dir = profiles_root / "local"
    examples_dir = profiles_root / "examples"
    local_dir.mkdir(parents=True)
    examples_dir.mkdir(parents=True)

    (examples_dir / "base.toml").write_text(
        """
[format.extensions.radio_discourse]
enabled = true
label_roles = false

[postprocess.extensions.radio_discourse]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    (local_dir / "child.toml").write_text(
        """
extends = ["base"]

[format.extensions.radio_discourse]
label_roles = true
window_size = 6
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profiles_root
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["format"]["extensions"]["radio_discourse"] == {
            "enabled": True,
            "label_roles": True,
            "window_size": 6,
        }
        assert data["postprocess"]["extensions"]["radio_discourse"] == {"enabled": True}
    finally:
        autosub.core.profile.Path = original_path


def test_corners_inherited_from_base_profile(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "base.toml").write_text(
        """
[[corners]]
name = "Fan Letter Corner"
description = "Reading fan letters"
cues = ["おたよりコーナー"]

[[corners]]
name = "Song Watchalong"
description = "Watching the MV"
cues = ["MVを見ていきましょう"]
""".strip(),
        encoding="utf-8",
    )
    (profile_dir / "child.toml").write_text(
        """
extends = ["base"]

[[corners]]
name = "Card Illustrations"
description = "Reviewing card art"
cues = ["カードイラスト"]
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        corners = data["corners"]
        assert len(corners) == 3
        names = [c["name"] for c in corners]
        # Base corners come first, then child's own
        assert names == ["Fan Letter Corner", "Song Watchalong", "Card Illustrations"]
    finally:
        autosub.core.profile.Path = original_path


def test_corners_inherited_through_chain(tmp_path):
    """Corners accumulate through a 3-level extends chain."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "grandparent.toml").write_text(
        """
[[corners]]
name = "Opening"
description = "Show opening"
cues = ["始まり"]
""".strip(),
        encoding="utf-8",
    )
    (profile_dir / "parent.toml").write_text(
        """
extends = ["grandparent"]

[[corners]]
name = "Middle"
description = "Middle segment"
cues = ["中盤"]
""".strip(),
        encoding="utf-8",
    )
    (profile_dir / "child.toml").write_text(
        """
extends = ["parent"]

[[corners]]
name = "Ending"
description = "Show ending"
cues = ["終わり"]
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        corners = data["corners"]
        assert len(corners) == 3
        names = [c["name"] for c in corners]
        assert names == ["Opening", "Middle", "Ending"]
    finally:
        autosub.core.profile.Path = original_path


def test_normalizer_terms_are_accumulated_across_profile_extends(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "base.toml").write_text(
        """
[format.normalizer]
engine = "llm"

[[format.normalizer.terms]]
value = "鈴原希実"
explanation = "Host name."
""".strip(),
        encoding="utf-8",
    )
    (profile_dir / "child.toml").write_text(
        """
extends = ["base"]

[format.normalizer]
provider = "google-vertex"

[[format.normalizer.terms]]
value = "のんばんは"
explanation = "Show greeting."
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("child")
        assert data["format"]["normalizer"] == {
            "engine": "llm",
            "provider": "google-vertex",
            "terms": [
                {"value": "鈴原希実", "explanation": "Host name."},
                {"value": "のんばんは", "explanation": "Show greeting."},
            ],
        }
    finally:
        autosub.core.profile.Path = original_path


def test_normalizer_keywords_shorthand_is_normalized_to_terms(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "keywords.toml").write_text(
        """
[format.normalizer]
engine = "llm"
keywords = ["鈴原希実", "のんばんは"]
""".strip(),
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("keywords")
        assert data["format"]["normalizer"] == {
            "engine": "llm",
            "terms": [{"value": "鈴原希実"}, {"value": "のんばんは"}],
        }
    finally:
        autosub.core.profile.Path = original_path


def test_no_corners_returns_empty_list(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "nocorners.toml").write_text(
        '[translate]\nprompt = "test"\n', encoding="utf-8"
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        data = load_unified_profile("nocorners")
        assert data["corners"] == []
    finally:
        autosub.core.profile.Path = original_path


def test_missing_profile_raises_file_not_found(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        with pytest.raises(FileNotFoundError, match="proseka/mmj.toml not found"):
            load_unified_profile("proseka/mmj")
    finally:
        autosub.core.profile.Path = original_path


def test_missing_extends_parent_raises_file_not_found(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "child.toml").write_text(
        'extends = ["does_not_exist"]\n', encoding="utf-8"
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        with pytest.raises(FileNotFoundError, match="does_not_exist.toml not found"):
            load_unified_profile("child")
    finally:
        autosub.core.profile.Path = original_path


def test_missing_prompt_file_raises_file_not_found(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "with_missing_prompt.toml").write_text(
        '[translate]\nprompt = "prompts/does_not_exist.md"\n', encoding="utf-8"
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts":
                return tmp_path / "prompts_empty"
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        with pytest.raises(
            FileNotFoundError, match="prompts/does_not_exist.md.*not found"
        ):
            load_unified_profile("with_missing_prompt")
    finally:
        autosub.core.profile.Path = original_path


def test_prompt_file_reference_emits_deprecation_warning(tmp_path, caplog):
    profile_dir = tmp_path / "profiles"
    prompts_dir = tmp_path / "prompts"
    profile_dir.mkdir()
    prompts_dir.mkdir()

    (profile_dir / "uses_prompt_file.toml").write_text(
        '[translate]\nprompt = "prompts/legacy.md"\n', encoding="utf-8"
    )
    (prompts_dir / "legacy.md").write_text(
        "Legacy prompt content from a file.", encoding="utf-8"
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            if args and args[0] == "prompts":
                return prompts_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        with caplog.at_level("WARNING", logger="autosub.core.profile"):
            data = load_unified_profile("uses_prompt_file")
        assert data["prompt"] == ["Legacy prompt content from a file."]
        assert "DEPRECATED" in caplog.text
        assert "prompts/legacy.md" in caplog.text
    finally:
        autosub.core.profile.Path = original_path


def test_inline_prompt_strings_do_not_emit_deprecation_warning(tmp_path, caplog):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    (profile_dir / "inline_only.toml").write_text(
        '[translate]\nprompt = ["Be concise.", "Use casual register."]\n',
        encoding="utf-8",
    )

    import autosub.core.profile

    original_path = autosub.core.profile.Path

    class MockPath(autosub.core.profile.Path):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "profiles":
                return profile_dir
            return super().__new__(cls, *args, **kwargs)

    autosub.core.profile.Path = MockPath

    try:
        with caplog.at_level("WARNING", logger="autosub.core.profile"):
            data = load_unified_profile("inline_only")
        assert data["prompt"] == ["Be concise.", "Use casual register."]
        assert "DEPRECATED" not in caplog.text
    finally:
        autosub.core.profile.Path = original_path
