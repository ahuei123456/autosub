import os
from pathlib import Path

import pytest

from autosub.core import ffmpeg_bin


# On Windows, shutil.which() respects PATHEXT and will not resolve an
# extensionless file. Give every binary the right suffix per-platform so the
# PATH-lookup tests exercise real behavior on both POSIX and Windows.
_EXE_SUFFIX = ".exe" if os.name == "nt" else ""


def _same_path(a: str, b: str) -> bool:
    """Compare paths in a way that survives Windows case-folding.

    shutil.which() on Windows returns the extension from PATHEXT (e.g.
    ``ffmpeg.EXE``) regardless of how the file is named on disk, so direct
    string equality fails even when both paths refer to the same file.
    """
    return os.path.normcase(a) == os.path.normcase(b)


@pytest.fixture
def isolated_path(tmp_path, monkeypatch):
    """Empty PATH and clear env vars so each test sees a clean slate."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.delenv("FFPROBE_PATH", raising=False)
    # Neutralize the Windows fallback dirs so a real ffmpeg install on the
    # host machine can't leak into the resolution chain during tests.
    monkeypatch.setattr(ffmpeg_bin, "_WINDOWS_FALLBACK_DIRS", ())
    return tmp_path


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_env_var_override_takes_precedence(isolated_path, monkeypatch):
    on_path = _make_executable(isolated_path / f"ffmpeg{_EXE_SUFFIX}")
    override = _make_executable(isolated_path / f"custom_ffmpeg{_EXE_SUFFIX}")
    monkeypatch.setenv("FFMPEG_PATH", str(override))

    resolved = ffmpeg_bin._resolve("ffmpeg", "FFMPEG_PATH")

    assert resolved == str(override)
    # PATH binary should NOT be selected when override exists
    assert not _same_path(resolved, str(on_path))


def test_resolves_via_path_when_no_env_override(isolated_path):
    on_path = _make_executable(isolated_path / f"ffmpeg{_EXE_SUFFIX}")
    resolved = ffmpeg_bin._resolve("ffmpeg", "FFMPEG_PATH")
    assert _same_path(resolved, str(on_path))


def test_falls_back_to_bare_name_when_nothing_found(isolated_path):
    # Empty PATH, no env, no Windows fallback dirs exist
    resolved = ffmpeg_bin._resolve("ffmpeg", "FFMPEG_PATH")
    assert resolved == "ffmpeg"


def test_env_var_override_works_even_when_path_lookup_fails(
    isolated_path, monkeypatch
):
    override = _make_executable(isolated_path / f"custom_ffmpeg{_EXE_SUFFIX}")
    monkeypatch.setenv("FFMPEG_PATH", str(override))
    # No "ffmpeg" on PATH — env var must win
    resolved = ffmpeg_bin._resolve("ffmpeg", "FFMPEG_PATH")
    assert resolved == str(override)


def test_module_level_constants_resolve_to_strings():
    # Whatever the resolution chain produces, both must be non-empty strings
    assert isinstance(ffmpeg_bin.FFMPEG_BIN, str) and ffmpeg_bin.FFMPEG_BIN
    assert isinstance(ffmpeg_bin.FFPROBE_BIN, str) and ffmpeg_bin.FFPROBE_BIN


@pytest.mark.skipif(os.name != "nt", reason="Windows-only fallback dirs")
def test_windows_fallback_dirs_searched(isolated_path, tmp_path, monkeypatch):
    fake_dir = tmp_path / "fake_program_files"
    fake_dir.mkdir()
    binary = _make_executable(fake_dir / "ffmpeg.exe")
    monkeypatch.setattr(
        ffmpeg_bin, "_WINDOWS_FALLBACK_DIRS", (str(fake_dir),)
    )
    resolved = ffmpeg_bin._resolve("ffmpeg", "FFMPEG_PATH")
    assert _same_path(resolved, str(binary))
