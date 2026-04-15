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
    backend: Literal["chirp_2", "whisperx"] | None = None
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


class SubtitleLine(BaseModel):
    text: str
    start_time: float
    end_time: float
    speaker: str | None = None
    role: str | None = None
    corner: str | None = None
    words: List[TranscribedWord] = Field(default_factory=list)
    replacement_spans: List[ReplacementSpan] = Field(default_factory=list)
