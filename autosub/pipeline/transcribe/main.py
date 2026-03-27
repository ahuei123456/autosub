from pathlib import Path
import logging
from typing import Any, cast
from uuid import uuid4

from autosub.pipeline.transcribe import audio, gcs, api
from autosub.core.schemas import TranscribedWord, TranscriptionResult
from autosub.core.config import GCS_BUCKET, PROJECT_ID
from autosub.core.utils import parse_timestamp

logger = logging.getLogger(__name__)


def _duration_seconds(value: Any) -> float:
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(cast(Any, total_seconds()))

    seconds = float(getattr(value, "seconds", 0))
    nanos = float(getattr(value, "nanos", 0))
    return seconds + (nanos / 1_000_000_000)


def transcribe(
    video_path: Path,
    output_json_path: Path,
    language_code: str = "ja-JP",
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> TranscriptionResult:
    """
    End-to-end transcription of a video file:
    1. Extracts audio
    2. Decides whether to use GCS (for files > 1min) or direct API (<= 1min)
    3. Calls Chirp 3 API
    4. Parses results and saves to disk
    """
    if not PROJECT_ID:
        raise ValueError("AUTOSUB_PROJECT_ID is not set in the environment.")

    logger.info(f"Extracting audio from {video_path}...")
    audio_path = audio.extract_audio(video_path, start_time, end_time)

    duration = audio.get_audio_duration(audio_path)
    logger.info(f"Audio duration: {duration:.2f} seconds")

    words_data = []
    offset = parse_timestamp(start_time) if start_time else 0.0

    try:
        if duration > 60:
            if not GCS_BUCKET:
                raise ValueError(
                    "AUTOSUB_GCS_BUCKET environment variable must be set for videos longer than 1 minute."
                )

            # Long-running GCS workflow
            gcs_dest = f"autosub_staging/{uuid4()}_{audio_path.name}"
            logger.info(f"Uploading audio to {gcs_dest}...")
            gcs_uri = gcs.upload_to_gcs(GCS_BUCKET, audio_path, gcs_dest)

            try:
                response = api.transcribe_uri(
                    gcs_uri, PROJECT_ID, language_code, vocabulary, num_speakers
                )

                with open("response_batch.json", "w", encoding="utf-8") as f:
                    f.write(str(response.results[gcs_uri]))

                # Parse Google's Batch response
                for result in response.results[
                    gcs_uri
                ].inline_result.transcript.results:
                    for alt in result.alternatives:
                        for w in alt.words:
                            words_data.append(
                                TranscribedWord(
                                    word=w.word,
                                    start_time=_duration_seconds(w.start_offset)
                                    + offset,
                                    end_time=_duration_seconds(w.end_offset) + offset,
                                    speaker=w.speaker_label
                                    if hasattr(w, "speaker_label")
                                    else None,
                                )
                            )
            finally:
                logger.info(f"Cleaning up GCS staging file {gcs_uri}...")
                gcs.delete_from_gcs(GCS_BUCKET, gcs_uri)

        else:
            # Fast synchronous workflow
            with open(audio_path, "rb") as f:
                audio_content = f.read()

            response = api.transcribe_local_file(
                audio_content, PROJECT_ID, language_code, vocabulary, num_speakers
            )
            # Parse Google's standard response
            for result in response.results:
                for alt in result.alternatives:
                    for w in alt.words:
                        words_data.append(
                            TranscribedWord(
                                word=w.word,
                                start_time=_duration_seconds(w.start_offset) + offset,
                                end_time=_duration_seconds(w.end_offset) + offset,
                                speaker=w.speaker_label
                                if hasattr(w, "speaker_label")
                                else None,
                            )
                        )

    finally:
        # Always clean up the local extracted audio
        if audio_path.exists():
            audio_path.unlink()

    # Serialize to Pydantic and save
    final_result = TranscriptionResult(words=words_data)
    logger.info(
        f"Found {len(words_data)} transcribed words. Saving to {output_json_path}..."
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(final_result.model_dump_json(indent=2))

    return final_result
