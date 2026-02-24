from pydantic import BaseModel
from typing import List


class TranscribedWord(BaseModel):
    word: str
    start_time: float
    end_time: float
    speaker: str | None = None


class TranscriptionResult(BaseModel):
    words: List[TranscribedWord]


class SubtitleLine(BaseModel):
    text: str
    start_time: float
    end_time: float
    speaker: str | None = None
