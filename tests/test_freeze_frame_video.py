import pytest
from scripts import freeze_frame_video as script


def test_resolve_job_paths_creates_default_output_for_non_subtitle_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    paths = script.resolve_job_paths(
        input_video=input_video,
        subtitle=None,
        output=None,
        frame_image=None,
        upscaled_image=None,
    )

    assert paths.input_video == input_video.resolve()
    assert paths.output_video.suffix == ".mkv"
    assert "freeze_1080p" in paths.output_video.name
    assert paths.frame_image.suffix == ".png"
    assert paths.subtitle is None


def test_resolve_job_paths_creates_default_output_for_subtitle_mode(tmp_path):
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

    assert paths.output_video.suffix == ".mp4"
    assert paths.subtitle == subtitle.resolve()


def test_resolve_job_paths_enforces_correct_suffix_based_on_mode(tmp_path):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    # Non-subtitle mode requires .mkv
    with pytest.raises(ValueError, match="requires .mkv output"):
        script.resolve_job_paths(
            input_video=input_video,
            subtitle=None,
            output=tmp_path / "bad.mp4",
            frame_image=None,
            upscaled_image=None,
        )

    # Subtitle mode requires .mp4
    subtitle = tmp_path / "sample.ass"
    subtitle.write_text("[Script Info]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires .mp4 output"):
        script.resolve_job_paths(
            input_video=input_video,
            subtitle=subtitle,
            output=tmp_path / "bad.mkv",
            frame_image=None,
            upscaled_image=None,
        )


def test_escape_filter_path_prefers_relative_posix_path_when_possible(
    tmp_path, monkeypatch
):
    subtitle = tmp_path / "dir[name]" / "line,one;two's.ass"
    subtitle.parent.mkdir()
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    filter_value = script.escape_filter_path(subtitle)

    assert not filter_value.startswith(tmp_path.resolve().as_posix())
    assert "\\[" in filter_value
    assert "\\]" in filter_value
    assert "\\," in filter_value
    assert "\\;" in filter_value
    assert "\\'" in filter_value
    assert "\\:" not in filter_value


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

    command = script.build_final_command(
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
    assert paths.output_video.as_posix() in rendered

    inputs = [command[i + 1] for i, v in enumerate(command) if v == "-i"]
    assert inputs[0] == paths.input_video.as_posix()
    assert paths.input_video.as_posix() in inputs
    assert paths.upscaled_image.as_posix() in inputs


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

    command = script.build_final_command(
        paths=paths,
        overwrite=True,
        fps=30,
        crf=18,
        preset="medium",
        audio_bitrate="192k",
    )
    rendered = " ".join(command)

    assert "subtitles=filename='" in rendered
    assert "-movflags" in command
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert paths.output_video.as_posix() in rendered


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


def test_parse_frame_rate_handles_fractional_rate():
    assert script.parse_frame_rate("30000/1001") == pytest.approx(29.97003, rel=1e-6)
    assert script.parse_frame_rate("24/1") == 24.0
    assert script.parse_frame_rate(None) == 0.0


def test_get_video_fps_prefers_avg_frame_rate(tmp_path, monkeypatch):
    input_video = tmp_path / "sample.webm"
    input_video.write_bytes(b"video")

    def fake_probe(_: str):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "avg_frame_rate": "24000/1001",
                    "r_frame_rate": "30/1",
                }
            ]
        }

    monkeypatch.setattr(script.ffmpeg, "probe", fake_probe)

    assert script.get_video_fps(input_video) == pytest.approx(23.9760239, rel=1e-6)
