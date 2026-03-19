import json
import logging
from pathlib import Path

from autosub.core.schemas import TranscriptionResult
from autosub.pipeline.format import chunker
from autosub.pipeline.format import generator
from autosub.pipeline.format.layout import wrap_subtitle_lines
from autosub.pipeline.format.timing import apply_timing_rules

logger = logging.getLogger(__name__)


def format_subtitles(
    input_json_path: Path,
    output_ass_path: Path,
    keyframes: list[int] | None = None,
    video_duration_ms: int | None = None,
    timing_config: dict | None = None,
) -> None:
    """
    Reads a transcript.json file, chunks the transcribed words into semantic lines,
    applies timing rules (gap snapping, min duration, keyframes),
    and generates an output .ass subtitle file.
    """
    if not input_json_path.exists():
        raise FileNotFoundError(f"Transcript JSON file not found: {input_json_path}")

    logger.info(f"Loading transcript from {input_json_path}...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate against Pydantic schema
    transcript = TranscriptionResult(**data)

    logger.info("Chunking transcript into semantic subtitle lines...")
    lines = chunker.chunk_words_to_lines(transcript.words)
    logger.info(f"Generated {len(lines)} subtitle lines.")

    logger.info("Applying timing rules (snapping, keyframes, min duration)...")
    if not timing_config:
        timing_config = {}

    lines = apply_timing_rules(
        lines,
        keyframes_ms=keyframes,
        video_duration_ms=video_duration_ms,
        min_duration_ms=timing_config.get("min_duration_ms", 500),
        snap_threshold_ms=timing_config.get("snap_threshold_ms", 250),
        conditional_snap_threshold_ms=timing_config.get(
            "conditional_snap_threshold_ms", 500
        ),
    )

    logger.info("Applying subtitle layout rules (max visible lines, wrapping)...")
    lines = wrap_subtitle_lines(
        lines,
        max_line_width=timing_config.get("max_line_width", 22),
        max_lines_per_subtitle=timing_config.get("max_lines_per_subtitle", 2),
    )

    logger.info(f"Writing .ass file to {output_ass_path}...")
    generator.generate_ass_file(lines, output_ass_path)
    logger.info("Subtitle formatting complete!")
