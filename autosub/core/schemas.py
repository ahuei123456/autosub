from typing import List, Literal, NamedTuple

from pydantic import BaseModel, Field


class TranscribedWord(BaseModel):
    word: str
    start_time: float
    end_time: float
    speaker: str | None = None
    confidence: float | None = None


class TranscriptionSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    words: List[TranscribedWord] = Field(default_factory=list)
    speaker: str | None = None
    confidence: float | None = None
    kind: Literal["segment", "sentence", "chunk", "result"] | None = None


class TranscriptionMetadata(BaseModel):
    backend: Literal["chirp_2", "chirp_3", "whisperx"] | None = None
    language: str | None = None
    model: str | None = None


class TranscriptionResult(BaseModel):
    words: List[TranscribedWord] = Field(default_factory=list)
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    metadata: TranscriptionMetadata | None = None


class ReplacementSpan(NamedTuple):
    orig_start: int
    orig_end: int
    replaced_start: int
    replaced_end: int


class TimedSubtitleFields(BaseModel):
    """Shared timing and metadata fields for render lines and pipeline cues."""

    start_time: float
    end_time: float
    speaker: str | None = None
    role: str | None = None
    corner: str | None = None
    words: List[TranscribedWord] = Field(default_factory=list)
    replacement_spans: List[ReplacementSpan] = Field(default_factory=list)


class SubtitleLine(TimedSubtitleFields):
    text: str


class SubtitleMetadata(BaseModel):
    source_transcripts: List[str] = Field(default_factory=list)
    transcribe_metadata: List[TranscriptionMetadata] = Field(default_factory=list)


class SubtitleCue(TimedSubtitleFields):
    id: str
    source_text: str
    normalized_source_text: str | None = None
    translated_text: str | None = None
    final_text: str | None = None


class SubtitleDocument(BaseModel):
    """Versioned pipeline document.

    Cue IDs are stable within one pipeline run, but not guaranteed across
    runs or future cue insertion/resegmentation.
    """

    schema_version: Literal[1] = 1
    stage: Literal["formatted", "translated", "postprocessed"]
    metadata: SubtitleMetadata = Field(default_factory=SubtitleMetadata)
    cues: List[SubtitleCue] = Field(default_factory=list)
