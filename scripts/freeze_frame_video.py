from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ffmpeg
from better_ffmpeg_progress import FfmpegLogLevel, FfmpegProcess

TARGET_RESOLUTION = (1920, 1080)
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
        type=float,
        default=None,
        help=(
            "Output video FPS for the still-image stream. "
            "Defaults to the source video FPS."
        ),
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
    def _abs(p: Path) -> Path:
        return p.expanduser().resolve()

    input_video = _abs(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    subtitle_path = _abs(subtitle) if subtitle else None
    if subtitle_path and not subtitle_path.is_file():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    parent = input_video.parent
    stem = input_video.stem
    ext = ".mp4" if subtitle_path else ".mkv"

    resolved_output = _abs(output) if output else parent / f"{stem}_freeze_1080p{ext}"
    resolved_frame = _abs(frame_image) if frame_image else parent / f"{stem}_frame0.png"
    resolved_upscaled = (
        _abs(upscaled_image) if upscaled_image else parent / f"{stem}_frame0_1080p.png"
    )

    if resolved_output.suffix.lower() != ext:
        mode = "Subtitle burn-in" if subtitle_path else "Non-subtitle"
        raise ValueError(f"{mode} mode requires {ext} output: {resolved_output}")

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


def parse_frame_rate(rate: str | None) -> float:
    if not rate:
        return 0.0

    if "/" in rate:
        num_text, den_text = rate.split("/", 1)
        num = float(num_text)
        den = float(den_text)
        if den == 0:
            return 0.0
        return num / den

    return float(rate)


def get_video_fps(video_path: Path) -> float:
    try:
        probe = ffmpeg.probe(video_path.as_posix())
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(
            f"Could not probe source video FPS for {video_path}: {stderr or exc}"
        ) from exc

    video_stream = next(
        (
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if not video_stream:
        raise RuntimeError(f"No video stream found in {video_path}")

    for field in ("avg_frame_rate", "r_frame_rate"):
        fps = parse_frame_rate(video_stream.get(field))
        if fps > 0:
            return fps

    raise RuntimeError(f"Could not determine source video FPS for {video_path}")


def run_ffmpeg(stream: Any, overwrite: bool = False) -> None:
    if overwrite:
        stream = ffmpeg.overwrite_output(stream)
    stream = stream.global_args("-hide_banner", "-loglevel", "error")
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(stderr or "ffmpeg command failed.") from exc


def ensure_output_paths(paths: JobPaths, overwrite: bool) -> None:
    if overwrite:
        return
    for p in (paths.frame_image, paths.upscaled_image, paths.output_video):
        if p.exists():
            raise FileExistsError(f"File exists (use --overwrite): {p}")


def escape_filter_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        filter_path = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        filter_path = resolved.as_posix()

    escaped = filter_path
    for char in ("\\", ":", "'", "[", "]", ",", ";"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def extract_first_frame(paths: JobPaths, overwrite: bool) -> None:
    stream = ffmpeg.input(paths.input_video.as_posix()).output(
        paths.frame_image.as_posix(), vframes=1, map="0:v:0"
    )
    run_ffmpeg(stream, overwrite=overwrite)


def upscale_frame(paths: JobPaths, overwrite: bool) -> None:
    width, height = TARGET_RESOLUTION
    stream = (
        ffmpeg.input(paths.frame_image.as_posix())
        .filter(
            "scale",
            width,
            height,
            flags="lanczos",
            force_original_aspect_ratio="decrease",
        )
        .filter("pad", width, height, "(ow-iw)/2", "(oh-ih)/2")
        .output(paths.upscaled_image.as_posix(), vframes=1)
    )
    run_ffmpeg(stream, overwrite=overwrite)


def build_final_command(
    paths: JobPaths,
    fps: float,
    crf: int,
    preset: str,
    audio_bitrate: str,
    overwrite: bool,
) -> list[str]:
    command = ["ffmpeg"]
    if overwrite:
        command.append("-y")

    command.extend(
        [
            "-i",
            paths.input_video.as_posix(),
            "-framerate",
            str(fps),
            "-loop",
            "1",
            "-i",
            paths.upscaled_image.as_posix(),
        ]
    )

    if paths.subtitle:
        command.extend(
            [
                "-filter_complex",
                f"[1:v]subtitles=filename='{escape_filter_path(paths.subtitle)}'[v]",
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

    command.append(paths.output_video.as_posix())
    return command


def create_video(
    paths: JobPaths,
    overwrite: bool,
    fps: float,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    command = build_final_command(paths, fps, crf, preset, audio_bitrate, overwrite)
    log_path = paths.output_video.with_name(f"{paths.output_video.stem}_ffmpeg.log")

    process = FfmpegProcess(
        command=command,
        ffmpeg_log_level=FfmpegLogLevel.ERROR,
        ffmpeg_log_file=log_path,
    )
    if process.run() != 0:
        raise RuntimeError(f"FFmpeg encode failed. Log: {log_path}")


def main() -> int:
    args = parse_args()
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if args.crf < 0:
        raise ValueError("--crf must be 0 or greater.")

    require_ffmpeg()
    paths = resolve_job_paths(
        args.input_video,
        args.subtitle,
        args.output,
        args.frame_image,
        args.upscaled_image,
    )
    ensure_output_paths(paths, args.overwrite)
    fps = float(args.fps) if args.fps is not None else get_video_fps(paths.input_video)

    steps = [
        ("Extracting frame", lambda: extract_first_frame(paths, args.overwrite)),
        ("Upscaling frame", lambda: upscale_frame(paths, args.overwrite)),
        (
            "Encoding video",
            lambda: create_video(
                paths,
                args.overwrite,
                fps,
                args.crf,
                args.preset,
                args.audio_bitrate,
            ),
        ),
    ]

    for msg, func in steps:
        print(f"{msg}...")
        func()

    print(f"Created: {paths.output_video}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
