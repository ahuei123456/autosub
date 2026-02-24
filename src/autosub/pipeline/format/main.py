import json
import logging
from pathlib import Path

from autosub.core.schemas import TranscriptionResult
from autosub.pipeline.format import chunker
from autosub.pipeline.format import generator

logger = logging.getLogger(__name__)


def format_subtitles(input_json_path: Path, output_ass_path: Path) -> None:
    """
    Reads a transcript.json file, chunks the transcribed words into semantic lines,
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

    logger.info(f"Writing .ass file to {output_ass_path}...")
    generator.generate_ass_file(lines, output_ass_path)
    logger.info("Subtitle formatting complete!")
