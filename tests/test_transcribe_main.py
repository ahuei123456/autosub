from pathlib import Path
from threading import Barrier
import time
from types import SimpleNamespace

from autosub.pipeline.transcribe import main as transcribe_main


def _offset(seconds: float) -> SimpleNamespace:
    whole_seconds = int(seconds)
    nanos = int(round((seconds - whole_seconds) * 1_000_000_000))
    return SimpleNamespace(seconds=whole_seconds, nanos=nanos)


def _make_local_response(
    word: str, start_time: float, end_time: float
) -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        words=[
                            SimpleNamespace(
                                word=word,
                                start_offset=_offset(start_time),
                                end_offset=_offset(end_time),
                            )
                        ]
                    )
                ]
            )
        ]
    )


def test_transcribe_merges_multiple_ranges_concurrently(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    output_path = tmp_path / "transcript.json"
    created_audio_paths: list[Path] = []
    barrier = Barrier(2)

    monkeypatch.setattr(transcribe_main, "PROJECT_ID", "project-id")

    def fake_extract_audio(
        input_video_path: Path,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> Path:
        audio_path = tmp_path / f"audio_{start_time}_{end_time}.wav"
        audio_path.write_text(start_time or "full", encoding="utf-8")
        created_audio_paths.append(audio_path)
        return audio_path

    monkeypatch.setattr(transcribe_main.audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        transcribe_main.audio, "get_audio_duration", lambda audio_path: 5.0
    )

    def fake_transcribe_local_file(
        audio_content: bytes,
        project_id: str,
        language_code: str = "ja-JP",
        vocabulary: list[str] | None = None,
        num_speakers: int | None = None,
    ) -> SimpleNamespace:
        segment_id = audio_content.decode("utf-8")
        barrier.wait(timeout=2)
        if segment_id == "0":
            time.sleep(0.1)
        return _make_local_response(f"word-{segment_id}", 0.1, 0.3)

    monkeypatch.setattr(
        transcribe_main.api, "transcribe_local_file", fake_transcribe_local_file
    )

    result = transcribe_main.transcribe(
        video_path,
        output_path,
        time_ranges=[("15", "20"), ("0", "5")],
    )

    assert [word.word for word in result.words] == ["word-0", "word-15"]
    assert [word.start_time for word in result.words] == [0.1, 15.1]
    assert output_path.exists()
    assert all(not path.exists() for path in created_audio_paths)


def test_transcribe_fails_if_any_segment_fails(tmp_path, monkeypatch):
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake", encoding="utf-8")
    output_path = tmp_path / "transcript.json"
    created_audio_paths: list[Path] = []
    barrier = Barrier(2)

    monkeypatch.setattr(transcribe_main, "PROJECT_ID", "project-id")

    def fake_extract_audio(
        input_video_path: Path,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> Path:
        audio_path = tmp_path / f"audio_{start_time}_{end_time}.wav"
        audio_path.write_text(start_time or "full", encoding="utf-8")
        created_audio_paths.append(audio_path)
        return audio_path

    monkeypatch.setattr(transcribe_main.audio, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        transcribe_main.audio, "get_audio_duration", lambda audio_path: 5.0
    )

    def fake_transcribe_local_file(
        audio_content: bytes,
        project_id: str,
        language_code: str = "ja-JP",
        vocabulary: list[str] | None = None,
        num_speakers: int | None = None,
    ) -> SimpleNamespace:
        segment_id = audio_content.decode("utf-8")
        barrier.wait(timeout=2)
        if segment_id == "15":
            raise RuntimeError("segment boom")
        return _make_local_response(f"word-{segment_id}", 0.1, 0.3)

    monkeypatch.setattr(
        transcribe_main.api, "transcribe_local_file", fake_transcribe_local_file
    )

    try:
        transcribe_main.transcribe(
            video_path,
            output_path,
            time_ranges=[("0", "5"), ("15", "20")],
        )
        assert False, "Expected transcription to fail when one segment fails."
    except RuntimeError as exc:
        assert "segment 2" in str(exc)
        assert "segment boom" in str(exc)

    assert not output_path.exists()
    assert all(not path.exists() for path in created_audio_paths)
