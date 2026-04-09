import logging
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

logger = logging.getLogger(__name__)


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

    # Wait for the operation to complete (default 900s can be too short)
    response = operation.result(timeout=1800)
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
