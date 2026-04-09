import logging
import time
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

logger = logging.getLogger(__name__)

DEFAULT_BATCH_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_BATCH_HEARTBEAT_SECONDS = 60.0


def _operation_name(operation: object) -> str:
    raw_operation = getattr(operation, "operation", None)
    if raw_operation is None:
        return "<unknown>"
    name = getattr(raw_operation, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return "<unknown>"


def _format_elapsed_seconds(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"
    return f"{remaining_minutes}m {remaining_seconds:02d}s"


def _wait_for_batch_operation(
    operation: object,
    *,
    gcs_uri: str,
    poll_interval_seconds: float | None = None,
    heartbeat_seconds: float | None = None,
) -> speech_v2.BatchRecognizeResponse:
    if poll_interval_seconds is None:
        poll_interval_seconds = DEFAULT_BATCH_POLL_INTERVAL_SECONDS
    if heartbeat_seconds is None:
        heartbeat_seconds = DEFAULT_BATCH_HEARTBEAT_SECONDS

    operation_name = _operation_name(operation)
    started_at = time.monotonic()
    last_heartbeat_at = started_at

    logger.info(
        "Submitted Chirp 2 batch job %s for %s. Polling every %.0f seconds.",
        operation_name,
        gcs_uri,
        poll_interval_seconds,
    )

    while not bool(getattr(operation, "done")()):
        time.sleep(poll_interval_seconds)
        current_time = time.monotonic()
        if current_time - last_heartbeat_at >= heartbeat_seconds:
            logger.info(
                "Still waiting on Chirp 2 batch job %s for %s after %s.",
                operation_name,
                gcs_uri,
                _format_elapsed_seconds(current_time - started_at),
            )
            last_heartbeat_at = current_time

    elapsed = time.monotonic() - started_at
    logger.info(
        "Chirp 2 batch job %s for %s completed in %s.",
        operation_name,
        gcs_uri,
        _format_elapsed_seconds(elapsed),
    )

    try:
        return operation.result()  # type: ignore[return-value]
    except Exception as exc:
        raise RuntimeError(
            f"Chirp 2 batch transcription failed for {gcs_uri}: {exc}"
        ) from exc


_CHIRP_ENDPOINTS = {
    "chirp_2": "us-central1-speech.googleapis.com",
    "chirp_3": "us-speech.googleapis.com",
}
_CHIRP_LOCATIONS = {
    "chirp_2": "us-central1",
    "chirp_3": "us",
}


def transcribe_uri(
    gcs_uri: str,
    project_id: str,
    language_code: str = "ja-JP",
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    model: str = "chirp_2",
) -> speech_v2.BatchRecognizeResponse:
    """
    Sends a long-running batch transcription request using a GCS URI.
    Required for audio files longer than 1 minute.
    """
    endpoint = _CHIRP_ENDPOINTS.get(model, _CHIRP_ENDPOINTS["chirp_2"])
    location = _CHIRP_LOCATIONS.get(model, _CHIRP_LOCATIONS["chirp_2"])
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=endpoint)
    )

    features = speech_v2.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=[language_code],
        model=model,
        features=features,
    )

    # SpeechAdaptation is only supported on Chirp 2
    if vocabulary and model == "chirp_2":
        config.adaptation = cloud_speech.SpeechAdaptation(
            phrase_sets=[
                cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=cloud_speech.PhraseSet(
                        phrases=[{"value": word} for word in vocabulary]
                    )
                )
            ]
        )
    elif vocabulary and model == "chirp_3":
        logger.debug(
            "Vocabulary hints ignored — Chirp 3 does not support SpeechAdaptation"
        )

    request = speech_v2.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
        config=config,
        files=[speech_v2.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=speech_v2.RecognitionOutputConfig(
            inline_response_config=speech_v2.InlineOutputConfig()
        ),
    )

    logger.info(f"Starting long-running transcription on {gcs_uri}...")
    operation = client.batch_recognize(request=request)
    response = _wait_for_batch_operation(operation, gcs_uri=gcs_uri)
    logger.info("Transcription complete!")
    return response  # type: ignore


def transcribe_local_file(
    audio_content: bytes,
    project_id: str,
    language_code: str = "ja-JP",
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    model: str = "chirp_2",
) -> speech_v2.RecognizeResponse:
    """
    Sends a standard synchronous transcription request using local audio bytes.
    Can only be used if audio is strictly < 1 minute.
    """
    endpoint = _CHIRP_ENDPOINTS.get(model, _CHIRP_ENDPOINTS["chirp_2"])
    location = _CHIRP_LOCATIONS.get(model, _CHIRP_LOCATIONS["chirp_2"])
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint=endpoint)
    )

    features = speech_v2.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=[language_code],
        model=model,
        features=features,
    )

    if vocabulary and model == "chirp_2":
        config.adaptation = cloud_speech.SpeechAdaptation(
            phrase_sets=[
                cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=cloud_speech.PhraseSet(
                        phrases=[{"value": word} for word in vocabulary]
                    )
                )
            ]
        )
    elif vocabulary and model == "chirp_3":
        logger.debug(
            "Vocabulary hints ignored — Chirp 3 does not support SpeechAdaptation"
        )

    request = speech_v2.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
        config=config,
        content=audio_content,
    )

    logger.info("Starting fast synchronous transcription...")
    response = client.recognize(request=request)
    logger.info("Transcription complete!")
    return response
