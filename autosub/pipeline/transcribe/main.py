from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence, cast
from uuid import uuid4

from autosub.core.config import GCS_BUCKET, PROJECT_ID
from autosub.core.schemas import TranscribedWord, TranscriptionResult
from autosub.core.utils import parse_timestamp
from autosub.pipeline.transcribe import api, audio, gcs

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TRANSCRIPTION_JOBS = 4


@dataclass(frozen=True)
class TimeRange:
    index: int
    start_time: str | None
    end_time: str | None
    offset_seconds: float


def _duration_seconds(value: Any) -> float:
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(cast(Any, total_seconds()))

    seconds = float(getattr(value, "seconds", 0))
    nanos = float(getattr(value, "nanos", 0))
    return seconds + (nanos / 1_000_000_000)


def _normalize_time_ranges(
    start_time: str | None,
    end_time: str | None,
    time_ranges: Sequence[tuple[str | None, str | None]] | None,
) -> list[TimeRange]:
    ranges = list(time_ranges) if time_ranges is not None else [(start_time, end_time)]
    if not ranges:
        ranges = [(None, None)]

    return [
        TimeRange(
            index=index,
            start_time=range_start,
            end_time=range_end,
            offset_seconds=parse_timestamp(range_start) if range_start else 0.0,
        )
        for index, (range_start, range_end) in enumerate(ranges)
    ]


def _parse_words(results: Any, offset_seconds: float) -> list[TranscribedWord]:
    words_data: list[TranscribedWord] = []
    for result in results:
        for alt in result.alternatives:
            for word_info in alt.words:
                words_data.append(
                    TranscribedWord(
                        word=word_info.word,
                        start_time=_duration_seconds(word_info.start_offset)
                        + offset_seconds,
                        end_time=_duration_seconds(word_info.end_offset)
                        + offset_seconds,
                        speaker=word_info.speaker_label
                        if hasattr(word_info, "speaker_label")
                        else None,
                    )
                )
    return words_data


def _transcribe_time_range(
    video_path: Path,
    project_id: str,
    language_code: str,
    vocabulary: list[str] | None,
    num_speakers: int | None,
    time_range: TimeRange,
) -> list[TranscribedWord]:
    logger.info(
        "Extracting audio for segment %s (start=%s, end=%s)...",
        time_range.index + 1,
        time_range.start_time,
        time_range.end_time,
    )
    audio_path = audio.extract_audio(
        video_path, time_range.start_time, time_range.end_time
    )

    try:
        duration = audio.get_audio_duration(audio_path)
        logger.info(
            "Segment %s audio duration: %.2f seconds",
            time_range.index + 1,
            duration,
        )

        if duration > 60:
            if not GCS_BUCKET:
                raise ValueError(
                    "AUTOSUB_GCS_BUCKET environment variable must be set for videos longer than 1 minute."
                )
            gcs_bucket = GCS_BUCKET

            gcs_dest = f"autosub_staging/{uuid4()}_{audio_path.name}"
            logger.info(
                "Uploading segment %s audio to %s...",
                time_range.index + 1,
                gcs_dest,
            )
            gcs_uri = gcs.upload_to_gcs(gcs_bucket, audio_path, gcs_dest)

            try:
                response = api.transcribe_uri(
                    gcs_uri, project_id, language_code, vocabulary, num_speakers
                )
                return _parse_words(
                    response.results[gcs_uri].inline_result.transcript.results,
                    time_range.offset_seconds,
                )
            finally:
                logger.info("Cleaning up GCS staging file %s...", gcs_uri)
                gcs.delete_from_gcs(gcs_bucket, gcs_uri)

        with audio_path.open("rb") as handle:
            audio_content = handle.read()

        response = api.transcribe_local_file(
            audio_content, project_id, language_code, vocabulary, num_speakers
        )
        return _parse_words(response.results, time_range.offset_seconds)
    finally:
        if audio_path.exists():
            audio_path.unlink()


def transcribe(
    video_path: Path,
    output_json_path: Path,
    language_code: str = "ja-JP",
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    time_ranges: Sequence[tuple[str | None, str | None]] | None = None,
) -> TranscriptionResult:
    """
    End-to-end transcription of a video file:
    1. Extracts audio for one or more segments
    2. Decides whether to use GCS (for files > 1min) or direct API (<= 1min)
    3. Calls the Chirp 2 API for each segment
    4. Merges results and saves to disk
    """
    if not PROJECT_ID:
        raise ValueError("AUTOSUB_PROJECT_ID is not set in the environment.")
    project_id = PROJECT_ID

    normalized_ranges = _normalize_time_ranges(start_time, end_time, time_ranges)
    logger.info(
        "Starting transcription for %s segment(s) from %s",
        len(normalized_ranges),
        video_path,
    )

    segment_results: dict[int, list[TranscribedWord]] = {}
    failures: list[tuple[int, Exception]] = []

    if len(normalized_ranges) == 1:
        segment = normalized_ranges[0]
        segment_results[segment.index] = _transcribe_time_range(
            video_path, project_id, language_code, vocabulary, num_speakers, segment
        )
    else:
        max_workers = min(MAX_CONCURRENT_TRANSCRIPTION_JOBS, len(normalized_ranges))
        logger.info(
            "Submitting %s transcription segment(s) with up to %s concurrent worker(s)...",
            len(normalized_ranges),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _transcribe_time_range,
                    video_path,
                    project_id,
                    language_code,
                    vocabulary,
                    num_speakers,
                    segment,
                ): segment
                for segment in normalized_ranges
            }

            for future in as_completed(futures):
                segment = futures[future]
                try:
                    segment_results[segment.index] = future.result()
                except Exception as exc:
                    failures.append((segment.index, exc))

    if failures:
        failures.sort(key=lambda item: item[0])
        failure_messages = ", ".join(
            f"segment {segment_index + 1}: {exc}" for segment_index, exc in failures
        )
        raise RuntimeError(
            f"One or more transcription segments failed: {failure_messages}"
        ) from failures[0][1]

    words_data: list[TranscribedWord] = []
    for segment in normalized_ranges:
        words_data.extend(segment_results.get(segment.index, []))

    words_data.sort(key=lambda word: (word.start_time, word.end_time))

    final_result = TranscriptionResult(words=words_data)
    logger.info(
        "Found %s transcribed words across %s segment(s). Saving to %s...",
        len(words_data),
        len(normalized_ranges),
        output_json_path,
    )

    with output_json_path.open("w", encoding="utf-8") as handle:
        handle.write(final_result.model_dump_json(indent=2))

    return final_result
