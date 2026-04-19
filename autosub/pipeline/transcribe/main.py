import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence, cast
from uuid import uuid4
from autosub.core.config import GCS_BUCKET, PROJECT_ID
from autosub.core.schemas import (
    TranscribedWord,
    TranscriptionMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)
from autosub.core.utils import parse_timestamp
from autosub.pipeline.transcribe import api, audio, gcs, whisperx_backend

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPTION_BACKEND = "chirp_2"
SUPPORTED_TRANSCRIPTION_BACKENDS = {
    DEFAULT_TRANSCRIPTION_BACKEND,
    "chirp_3",
    "whisperx",
}
CHIRP_BACKENDS = {"chirp_2", "chirp_3"}
MAX_CONCURRENT_TRANSCRIPTION_JOBS = 4
MAX_CONCURRENT_WHISPERX_JOBS = 1


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


def _clamp_word_timestamps(
    raw_start: float, raw_end: float, chunk_duration: float = 0.0
) -> tuple[float, float]:
    """Clamp bogus Chirp 3 word timestamps before applying any time offset.

    Chirp 3 internally chunks audio at 18-minute intervals and sometimes
    returns a previous chunk boundary as the endOffset for words in later
    chunks (e.g. end_time=1080 for a word at start_time=1853).  This
    produces end < start, which corrupts subtitle timing.

    When we do our own chunking and know the chunk_duration, we can also
    catch values that exceed it.  But even without chunk_duration, the
    end < start check catches the Chirp pattern.

    Must be called *before* adding any time offset — otherwise a bogus 0 s
    end becomes a plausible-looking timestamp after the offset is added.
    """
    start = raw_start
    end = raw_end

    if chunk_duration > 0:
        if start > chunk_duration:
            start = end if end <= chunk_duration else 0.0
        if end > chunk_duration:
            end = start

    if end <= 0 or end < start:
        end = start

    return start, end


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


def _parse_words(
    results: Any, offset_seconds: float, chunk_duration: float = 0.0
) -> list[TranscribedWord]:
    words_data: list[TranscribedWord] = []
    clamped_count = 0
    for result in results:
        for alt in result.alternatives:
            for word_info in alt.words:
                raw_start = _duration_seconds(word_info.start_offset)
                raw_end = _duration_seconds(word_info.end_offset)
                start, end = _clamp_word_timestamps(raw_start, raw_end, chunk_duration)
                if (start, end) != (raw_start, raw_end):
                    clamped_count += 1
                words_data.append(
                    TranscribedWord(
                        word=word_info.word,
                        start_time=start + offset_seconds,
                        end_time=end + offset_seconds,
                        speaker=word_info.speaker_label
                        if hasattr(word_info, "speaker_label")
                        else None,
                    )
                )
    if clamped_count:
        logger.warning(
            "Clamped %d bogus word timestamp(s) from Chirp API response.",
            clamped_count,
        )
    return words_data


def _segment_speaker(words: Sequence[TranscribedWord]) -> str | None:
    speakers = {word.speaker for word in words if word.speaker}
    if len(speakers) == 1:
        return next(iter(speakers))
    return None


def _segment_confidence(alternative: Any) -> float | None:
    confidence = getattr(alternative, "confidence", None)
    if confidence is None:
        return None
    return float(confidence)


def _parse_chirp_segments(
    results: Any, offset_seconds: float, chunk_duration: float = 0.0
) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    clamped_count = 0
    for result in results:
        alternatives = getattr(result, "alternatives", [])
        if not alternatives:
            continue
        alternative = alternatives[0]
        segment_words: list[TranscribedWord] = []
        for word_info in getattr(alternative, "words", []):
            raw_start = _duration_seconds(word_info.start_offset)
            raw_end = _duration_seconds(word_info.end_offset)
            start, end = _clamp_word_timestamps(raw_start, raw_end, chunk_duration)
            if (start, end) != (raw_start, raw_end):
                clamped_count += 1
            segment_words.append(
                TranscribedWord(
                    word=word_info.word,
                    start_time=start + offset_seconds,
                    end_time=end + offset_seconds,
                    speaker=(
                        word_info.speaker_label
                        if hasattr(word_info, "speaker_label")
                        else None
                    ),
                )
            )
        if segment_words:
            segment_start = segment_words[0].start_time
            segment_end = segment_words[-1].end_time
        else:
            continue
        segments.append(
            TranscriptionSegment(
                text=str(getattr(alternative, "transcript", "")).strip(),
                start_time=segment_start,
                end_time=segment_end,
                words=segment_words,
                speaker=_segment_speaker(segment_words),
                confidence=_segment_confidence(alternative),
                kind="result",
            )
        )
    if clamped_count:
        logger.warning(
            "Clamped %d bogus segment word timestamp(s) from Chirp API response.",
            clamped_count,
        )
    return segments


def _apply_offset(
    words: Sequence[TranscribedWord], offset_seconds: float
) -> list[TranscribedWord]:
    return [
        TranscribedWord(
            word=word.word,
            start_time=word.start_time + offset_seconds,
            end_time=word.end_time + offset_seconds,
            speaker=word.speaker,
            confidence=word.confidence,
        )
        for word in words
    ]


def _apply_offset_to_segments(
    segments: Sequence[TranscriptionSegment], offset_seconds: float
) -> list[TranscriptionSegment]:
    return [
        TranscriptionSegment(
            text=segment.text,
            start_time=segment.start_time + offset_seconds,
            end_time=segment.end_time + offset_seconds,
            words=_apply_offset(segment.words, offset_seconds),
            speaker=segment.speaker,
            confidence=segment.confidence,
            kind=segment.kind,
        )
        for segment in segments
    ]


def _validate_transcription_backend(transcription_backend: str) -> str:
    normalized = transcription_backend.strip().lower()
    if normalized not in SUPPORTED_TRANSCRIPTION_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_BACKENDS))
        raise ValueError(
            f"Unsupported transcription backend: {transcription_backend}. "
            f"Supported values: {supported}."
        )
    return normalized


def _transcribe_time_range(
    video_path: Path,
    project_id: str | None,
    language_code: str,
    vocabulary: list[str] | None,
    num_speakers: int | None,
    time_range: TimeRange,
    transcription_backend: str,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    whisper_batch_size: int,
    whisper_diarize: bool,
    whisper_hf_token: str | None,
) -> TranscriptionResult:
    logger.info(
        "Extracting audio for segment %s (start=%s, end=%s)...",
        time_range.index + 1,
        time_range.start_time,
        time_range.end_time,
    )
    audio_path = audio.extract_audio(
        video_path,
        time_range.start_time,
        time_range.end_time,
        opus=(transcription_backend == "chirp_3"),
    )

    try:
        duration = audio.get_audio_duration(audio_path)
        logger.info(
            "Segment %s audio duration: %.2f seconds",
            time_range.index + 1,
            duration,
        )

        if transcription_backend == "whisperx":
            whisper_result = whisperx_backend.transcribe_file(
                audio_path,
                language_code=language_code,
                model_name=whisper_model,
                device=whisper_device,
                compute_type=whisper_compute_type,
                batch_size=whisper_batch_size,
                diarize=whisper_diarize,
                hf_token=whisper_hf_token,
                num_speakers=num_speakers,
            )
            return TranscriptionResult(
                words=_apply_offset(whisper_result.words, time_range.offset_seconds),
                segments=_apply_offset_to_segments(
                    whisper_result.segments, time_range.offset_seconds
                ),
                metadata=whisper_result.metadata,
            )

        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in the environment.")
        project_id_str = project_id

        if duration > 60:
            if not GCS_BUCKET:
                raise ValueError(
                    "AUTOSUB_GCS_BUCKET environment variable must be set for videos longer than 1 minute."
                )
            gcs_bucket = GCS_BUCKET
            max_chunk_seconds = audio.MAX_CHUNK_MINUTES * 60
            needs_chunking = (
                transcription_backend == "chirp_3" and duration > max_chunk_seconds
            )

            if needs_chunking:
                # Split into chunks for Chirp 3's word-timestamp limit
                with tempfile.TemporaryDirectory() as tmp_dir:
                    chunks = audio.split_audio(
                        audio_path, max_chunk_seconds, Path(tmp_dir)
                    )
                    logger.info(
                        "Split audio into %d chunks (%d min each)",
                        len(chunks),
                        audio.MAX_CHUNK_MINUTES,
                    )

                    chunk_results: dict[
                        int, tuple[list[TranscribedWord], list[TranscriptionSegment]]
                    ] = {}
                    chunk_failures: list[tuple[int, Exception]] = []
                    max_workers = min(MAX_CONCURRENT_TRANSCRIPTION_JOBS, len(chunks))

                    def _transcribe_chunk(
                        chunk_idx: int,
                        chunk_path: Path,
                        chunk_start: float,
                    ) -> tuple[int, list[TranscribedWord], list[TranscriptionSegment]]:
                        chunk_duration = audio.get_audio_duration(chunk_path)
                        gcs_dest = f"autosub_staging/{uuid4()}_{chunk_path.name}"
                        logger.info(
                            "  Chunk %d/%d: uploading to %s...",
                            chunk_idx + 1,
                            len(chunks),
                            gcs_dest,
                        )
                        gcs_uri = gcs.upload_to_gcs(gcs_bucket, chunk_path, gcs_dest)
                        chunk_offset = chunk_start + time_range.offset_seconds
                        try:
                            response = api.transcribe_uri(
                                gcs_uri,
                                project_id_str,
                                language_code,
                                vocabulary,
                                num_speakers,
                                model=transcription_backend,
                            )
                            chirp_results = response.results[
                                gcs_uri
                            ].inline_result.transcript.results
                            return (
                                chunk_idx,
                                _parse_words(
                                    chirp_results, chunk_offset, chunk_duration
                                ),
                                _parse_chirp_segments(
                                    chirp_results, chunk_offset, chunk_duration
                                ),
                            )
                        finally:
                            logger.info("  Cleaning up %s...", gcs_dest)
                            gcs.delete_from_gcs(gcs_bucket, gcs_uri)

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(_transcribe_chunk, idx, path, start): idx
                            for idx, (path, start) in enumerate(chunks)
                        }
                        for future in as_completed(futures):
                            idx = futures[future]
                            try:
                                cidx, words, segs = future.result()
                                chunk_results[cidx] = (words, segs)
                            except Exception as exc:
                                chunk_failures.append((idx, exc))

                    if chunk_failures:
                        chunk_failures.sort(key=lambda item: item[0])
                        msgs = ", ".join(
                            f"chunk {i + 1}: {exc}" for i, exc in chunk_failures
                        )
                        raise RuntimeError(
                            f"One or more audio chunks failed: {msgs}"
                        ) from chunk_failures[0][1]

                    all_words: list[TranscribedWord] = []
                    all_segments: list[TranscriptionSegment] = []
                    for idx in sorted(chunk_results):
                        words, segs = chunk_results[idx]
                        all_words.extend(words)
                        all_segments.extend(segs)

                return TranscriptionResult(
                    words=all_words,
                    segments=all_segments,
                )

            # Single file — no chunking needed
            gcs_dest = f"autosub_staging/{uuid4()}_{audio_path.name}"
            logger.info(
                "Uploading segment %s audio to %s...",
                time_range.index + 1,
                gcs_dest,
            )
            gcs_uri = gcs.upload_to_gcs(gcs_bucket, audio_path, gcs_dest)

            try:
                response = api.transcribe_uri(
                    gcs_uri,
                    project_id_str,
                    language_code,
                    vocabulary,
                    num_speakers,
                    model=transcription_backend,
                )
                chirp_results = response.results[
                    gcs_uri
                ].inline_result.transcript.results
                return TranscriptionResult(
                    words=_parse_words(
                        chirp_results, time_range.offset_seconds, duration
                    ),
                    segments=_parse_chirp_segments(
                        chirp_results, time_range.offset_seconds, duration
                    ),
                )
            finally:
                logger.info("Cleaning up GCS staging file %s...", gcs_uri)
                gcs.delete_from_gcs(gcs_bucket, gcs_uri)

        with audio_path.open("rb") as handle:
            audio_content = handle.read()

        response = api.transcribe_local_file(
            audio_content,
            project_id_str,
            language_code,
            vocabulary,
            num_speakers,
            model=transcription_backend,
        )
        return TranscriptionResult(
            words=_parse_words(response.results, time_range.offset_seconds, duration),
            segments=_parse_chirp_segments(
                response.results, time_range.offset_seconds, duration
            ),
        )
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
    transcription_backend: str = DEFAULT_TRANSCRIPTION_BACKEND,
    whisper_model: str = "large-v2",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    whisper_batch_size: int = 16,
    whisper_diarize: bool = False,
    whisper_hf_token: str | None = None,
) -> TranscriptionResult:
    """
    End-to-end transcription of a video file:
    1. Extracts audio for one or more segments
    2. Decides whether to use GCS (for files > 1min) or direct API (<= 1min)
    3. Calls the selected Chirp model (chirp_2 or chirp_3) for each segment
    4. For chirp_3, splits audio into 18-min chunks to avoid empty results
    5. Merges results and saves to disk
    """
    resolved_backend = _validate_transcription_backend(transcription_backend)
    project_id = PROJECT_ID if resolved_backend in CHIRP_BACKENDS else None
    if resolved_backend == "whisperx" and vocabulary:
        logger.warning(
            "Ignoring vocabulary hints for WhisperX transcription; this backend "
            "does not support Chirp-style phrase adaptation."
        )

    normalized_ranges = _normalize_time_ranges(start_time, end_time, time_ranges)
    logger.info(
        "Starting %s transcription for %s segment(s) from %s",
        resolved_backend,
        len(normalized_ranges),
        video_path,
    )

    segment_results: dict[int, TranscriptionResult] = {}
    failures: list[tuple[int, Exception]] = []

    if len(normalized_ranges) == 1:
        segment = normalized_ranges[0]
        segment_results[segment.index] = _transcribe_time_range(
            video_path,
            project_id,
            language_code,
            vocabulary,
            num_speakers,
            segment,
            resolved_backend,
            whisper_model,
            whisper_device,
            whisper_compute_type,
            whisper_batch_size,
            whisper_diarize,
            whisper_hf_token,
        )
    else:
        concurrency_cap = (
            MAX_CONCURRENT_WHISPERX_JOBS
            if resolved_backend == "whisperx"
            else MAX_CONCURRENT_TRANSCRIPTION_JOBS
        )
        max_workers = min(concurrency_cap, len(normalized_ranges))
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
                    resolved_backend,
                    whisper_model,
                    whisper_device,
                    whisper_compute_type,
                    whisper_batch_size,
                    whisper_diarize,
                    whisper_hf_token,
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
    segments_data: list[TranscriptionSegment] = []
    metadata: TranscriptionMetadata | None = None
    for segment in normalized_ranges:
        segment_result = segment_results.get(segment.index)
        if segment_result is None:
            continue
        words_data.extend(segment_result.words)
        segments_data.extend(segment_result.segments)
        if metadata is None and segment_result.metadata is not None:
            metadata = segment_result.metadata

    words_data.sort(key=lambda word: (word.start_time, word.end_time))
    segments_data.sort(key=lambda segment: (segment.start_time, segment.end_time))

    final_result = TranscriptionResult(
        words=words_data,
        segments=segments_data,
        metadata=metadata
        or TranscriptionMetadata(
            backend=cast(Any, resolved_backend),
            language=language_code,
            model=(
                whisper_model if resolved_backend == "whisperx" else resolved_backend
            ),
        ),
    )
    logger.info(
        "Found %s transcribed words across %s segment(s). Saving to %s...",
        len(words_data),
        len(normalized_ranges),
        output_json_path,
    )

    with output_json_path.open("w", encoding="utf-8") as handle:
        handle.write(final_result.model_dump_json(indent=2))

    return final_result
