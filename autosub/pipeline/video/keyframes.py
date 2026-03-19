import shutil
import subprocess
from pathlib import Path
import ffmpeg


def check_dependencies() -> None:
    """Verifies that ffmpeg and SCXvid exist in the system PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in system PATH. Please install ffmpeg.")
    if not shutil.which("SCXvid.exe") and not shutil.which("SCXvid"):
        raise RuntimeError("SCXvid not found in system PATH. Please install SCXvid.exe")


def extract_keyframes(video_path: Path, output_path: Path) -> None:
    """Extracts keyframes from a video using ffmpeg and SCXvid."""
    check_dependencies()

    # We use subprocess directly to pipe ffmpeg output to SCXvid
    # ffmpeg -i %video% -f yuv4mpegpipe -vf scale=640:360 -pix_fmt yuv420p -vsync drop - | SCXvid.exe %video2%_keyframes.log
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-f",
        "yuv4mpegpipe",
        "-vf",
        "scale=640:360",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "drop",
        "-",
    ]

    scxvid_cmd = ["SCXvid", str(output_path)]

    try:
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        # SCXvid takes the yuv4mpegpipe on stdin
        subprocess.run(
            scxvid_cmd,
            stdin=ffmpeg_proc.stdout,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ffmpeg_proc.stdout:
            ffmpeg_proc.stdout.close()
        ffmpeg_proc.wait()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SCXvid command failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract keyframes: {e}")


def get_fps(video_path: Path) -> float:
    """Gets the exact frame rate of a video using ffprobe."""
    try:
        probe = ffmpeg.probe(str(video_path))
        video_stream = next(
            (
                stream
                for stream in probe.get("streams", [])
                if stream.get("codec_type") == "video"
            ),
            None,
        )
        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        # r_frame_rate is usually a fraction like "24000/1001" or "30/1"
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        num, den = map(int, r_frame_rate.split("/"))
        if den == 0:
            return 0.0
        return num / den
    except Exception as e:
        raise RuntimeError(f"Failed to extract framerate: {e}")


def parse_aegisub_keyframes(log_path: Path, fps: float) -> list[int]:
    """Parses an Aegisub keyframe log (.log) to a list of MS timestamps."""
    if fps <= 0:
        raise ValueError("FPS must be greater than 0")

    keyframes_ms: list[int] = []

    if not log_path.exists():
        return keyframes_ms

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore comments and headers
            if not line or line.startswith("#"):
                continue

            try:
                frame_number = int(line)
                # Convert frame number to MS
                timestamp_ms = int(frame_number * 1000 / fps)
                keyframes_ms.append(timestamp_ms)
            except ValueError:
                continue

    return keyframes_ms
