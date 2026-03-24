from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ffmpeg
from better_ffmpeg_progress import FfmpegLogLevel, FfmpegProcess

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
DEFAULT_OUTPUT_FPS = 30
DEFAULT_CRF = 18
DEFAULT_AUDIO_BITRATE = "192k"
DEFAULT_PRESET = "medium"


@dataclass(frozen=True)
class JobPaths:
    input_video: Path
    subtitle: Path | None
    frame_image: Path
    upscaled_image: Path
    output_video: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1080p still-image video from the first frame of an input video. "
            "Without subtitles, the output is MKV. With --subtitle, subtitles are "
            "burned in and the output is MP4."
        )
    )
    parser.add_argument("input_video", type=Path, help="Path to the source video.")
    parser.add_argument(
        "--subtitle",
        type=Path,
        help="Optional subtitle file to burn into the final MP4.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional final video path. Defaults to MKV or MP4 based on mode.",
    )
    parser.add_argument(
        "--frame-image",
        type=Path,
        help="Optional path for the extracted first-frame PNG.",
    )
    parser.add_argument(
        "--upscaled-image",
        type=Path,
        help="Optional path for the 1080p upscaled PNG.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_OUTPUT_FPS,
        help=f"Output video FPS for the still-image stream. Default: {DEFAULT_OUTPUT_FPS}.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help=f"libx264 CRF value. Default: {DEFAULT_CRF}.",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"libx264 preset. Default: {DEFAULT_PRESET}.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default=DEFAULT_AUDIO_BITRATE,
        help=f"AAC bitrate. Default: {DEFAULT_AUDIO_BITRATE}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def resolve_job_paths(
    input_video: Path,
    subtitle: Path | None,
    output: Path | None,
    frame_image: Path | None,
    upscaled_image: Path | None,
) -> JobPaths:
    input_video = input_video.expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    subtitle_path = subtitle.expanduser().resolve() if subtitle else None
    if subtitle_path and not subtitle_path.is_file():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    parent = input_video.parent
    stem = input_video.stem
    default_output_suffix = ".mp4" if subtitle_path else ".mkv"

    resolved_output = (
        output.expanduser().resolve()
        if output
        else parent / f"{stem}_freeze_1080p{default_output_suffix}"
    )
    resolved_frame = (
        frame_image.expanduser().resolve()
        if frame_image
        else parent / f"{stem}_frame0.png"
    )
    resolved_upscaled = (
        upscaled_image.expanduser().resolve()
        if upscaled_image
        else parent / f"{stem}_frame0_1080p.png"
    )

    expected_suffix = ".mp4" if subtitle_path else ".mkv"
    if resolved_output.suffix.lower() != expected_suffix:
        mode = "subtitle burn-in mode" if subtitle_path else "non-subtitle mode"
        raise ValueError(
            f"{mode} requires an output path ending in {expected_suffix}: {resolved_output}"
        )

    return JobPaths(
        input_video=input_video,
        subtitle=subtitle_path,
        frame_image=resolved_frame,
        upscaled_image=resolved_upscaled,
        output_video=resolved_output,
    )


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg was not found in PATH.")


def run_stream(stream: Any, overwrite: bool) -> None:
    if overwrite:
        stream = ffmpeg.overwrite_output(stream)

    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(stderr or "ffmpeg command failed.") from exc


def add_common_global_args(stream: Any) -> Any:
    return stream.global_args("-hide_banner", "-loglevel", "error")


def ensure_output_paths(paths: JobPaths, overwrite: bool) -> None:
    if overwrite:
        return

    for candidate in (paths.frame_image, paths.upscaled_image, paths.output_video):
        if candidate.exists():
            raise FileExistsError(
                f"Output already exists. Re-run with --overwrite to replace it: {candidate}"
            )


def escape_subtitles_filter_path(path: Path) -> str:
    escaped = path.resolve().as_posix()
    for char in ("\\", ":", "'", "[", "]", ",", ";"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def build_subtitles_filter(path: Path) -> str:
    return escape_subtitles_filter_path(path)


def extract_first_frame(paths: JobPaths, overwrite: bool) -> None:
    stream = ffmpeg.input(str(paths.input_video)).output(
        str(paths.frame_image),
        vframes=1,
        **{"map": "0:v:0"},
    )
    run_stream(add_common_global_args(stream), overwrite=overwrite)


def upscale_frame(paths: JobPaths, overwrite: bool) -> None:
    video = ffmpeg.input(str(paths.frame_image)).filter(
        "scale",
        TARGET_WIDTH,
        TARGET_HEIGHT,
        flags="lanczos",
        force_original_aspect_ratio="decrease",
    )
    video = video.filter(
        "pad",
        TARGET_WIDTH,
        TARGET_HEIGHT,
        "(ow-iw)/2",
        "(oh-ih)/2",
    )
    stream = ffmpeg.output(
        video,
        str(paths.upscaled_image),
        vframes=1,
    )
    run_stream(add_common_global_args(stream), overwrite=overwrite)


def build_video_command(
    paths: JobPaths,
    overwrite: bool,
    fps: int,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> list[str]:
    command = ["ffmpeg"]
    if overwrite:
        command.append("-y")

    command.extend(
        [
            "-i",
            str(paths.input_video),
            "-framerate",
            str(fps),
            "-loop",
            "1",
            "-i",
            str(paths.upscaled_image),
        ]
    )

    if paths.subtitle:
        command.extend(
            [
                "-filter_complex",
                f"[1:v]subtitles=filename={build_subtitles_filter(paths.subtitle)}[v]",
                "-map",
                "[v]",
            ]
        )
    else:
        command.extend(["-map", "1:v:0"])

    command.extend(
        [
            "-map",
            "0:a:0",
            "-vcodec",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-acodec",
            "aac",
            "-b:a",
            audio_bitrate,
            "-shortest",
        ]
    )

    if paths.subtitle:
        command.extend(["-movflags", "+faststart"])

    command.append(str(paths.output_video))
    return command


def create_video(
    paths: JobPaths,
    overwrite: bool,
    fps: int,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    command = build_video_command(
        paths=paths,
        overwrite=overwrite,
        fps=fps,
        crf=crf,
        preset=preset,
        audio_bitrate=audio_bitrate,
    )
    log_path = paths.output_video.with_name(f"{paths.output_video.stem}_ffmpeg.log")
    process = FfmpegProcess(
        command=command,
        ffmpeg_log_level=FfmpegLogLevel.ERROR,
        ffmpeg_log_file=log_path,
    )
    return_code = process.run()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg encode failed. Check log for details: {log_path}")


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if args.crf < 0:
        raise ValueError("--crf must be 0 or greater.")

    require_ffmpeg()
    paths = resolve_job_paths(
        input_video=args.input_video,
        subtitle=args.subtitle,
        output=args.output,
        frame_image=args.frame_image,
        upscaled_image=args.upscaled_image,
    )
    ensure_output_paths(paths, overwrite=args.overwrite)

    print("Extracting first frame...")
    extract_first_frame(paths, overwrite=args.overwrite)
    print("Upscaling frame to 1080p...")
    upscale_frame(paths, overwrite=args.overwrite)
    print("Encoding final video...")
    create_video(
        paths,
        overwrite=args.overwrite,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
    )
    print(f"Created {paths.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
