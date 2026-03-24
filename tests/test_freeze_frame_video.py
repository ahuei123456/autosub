from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "freeze_frame_video.py"
    spec = importlib.util.spec_from_file_location("freeze_frame_video", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = load_script_module()


def test_resolve_job_paths_defaults_to_mkv_without_subtitles(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    assert paths.frame_image == tmp_path / "sample_frame0.png"
    assert paths.upscaled_image == tmp_path / "sample_frame0_1080p.png"
    assert paths.output_video == tmp_path / "sample_freeze_1080p.mkv"


def test_resolve_job_paths_defaults_to_mp4_with_subtitles(tmp_path):
    input_video = tmp_path / "sample.webm"
    subtitle = tmp_path / "sample.ass"
    input_video.write_bytes(b"video")
    subtitle.write_text("[Script Info]\n", encoding="utf-8")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=subtitle,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    assert paths.output_video == tmp_path / "sample_freeze_1080p.mp4"
    assert paths.subtitle == subtitle


def test_resolve_job_paths_rejects_wrong_extension_for_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    with pytest.raises(ValueError, match=r"\.mkv"):
        script.resolve_job_paths(
            input_video=input_video,
            subtitle=None,
            output=tmp_path / "sample.mp4",
            frame_image=None,
            upscaled_image=None,
        )


def test_build_subtitles_filter_escapes_windows_style_path(tmp_path):
    subtitle = tmp_path / "dir[name]" / "line,one;two's.ass"
    subtitle.parent.mkdir()
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    filter_value = script.build_subtitles_filter(subtitle)

    assert "\\[" in filter_value
    assert "\\]" in filter_value
    assert "\\," in filter_value
    assert "\\;" in filter_value
    assert "\\'" in filter_value


def test_build_video_command_uses_requested_codecs(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    command = script.build_video_command(
        paths=paths,
        overwrite=True,
        fps=30,
        crf=18,
        preset="medium",
        audio_bitrate="192k",
    )
    rendered = " ".join(command)

    assert "-vcodec" in command
    assert command[command.index("-vcodec") + 1] == "libx264"
    assert "-acodec" in command
    assert command[command.index("-acodec") + 1] == "aac"
    assert "-b:a" in command
    assert command[command.index("-b:a") + 1] == "192k"
    assert str(paths.output_video) in rendered
    first_input_index = command.index("-i")
    assert command[first_input_index + 1] == str(paths.input_video)


def test_build_video_command_adds_subtitle_filter_for_burn_in_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    subtitle = tmp_path / "sample.ass"
    input_video.write_bytes(b"video")
    subtitle.write_text("[Script Info]\n", encoding="utf-8")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=subtitle,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    command = script.build_video_command(
        paths=paths,
        overwrite=True,
        fps=30,
        crf=18,
        preset="medium",
        audio_bitrate="192k",
    )
    rendered = " ".join(command)

    assert "subtitles=" in rendered
    assert "-movflags" in command
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert str(paths.output_video) in rendered


def test_ensure_output_paths_requires_overwrite_for_existing_files(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")
    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )
    paths.frame_image.write_bytes(b"png")

    with pytest.raises(FileExistsError, match="--overwrite"):
        script.ensure_output_paths(paths, overwrite=False)
