from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from autosub.core.errors import VertexResponseShapeError
from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)


class CornerDecision(BaseModel):
    id: int
    corner: str | None = None


class VertexCornerClassifier(BaseStructuredLLM):
    DEFAULT_MODELS = {
        "google-vertex": "gemini-3.1-flash-lite-preview",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
    }

    def __init__(
        self,
        *,
        project_id: str | None,
        segments: list[dict],
        model: str | None = None,
        location: str = "global",
        temperature: float = 0.1,
        provider: str = "google-vertex",
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
        reasoning_budget_tokens: int | None = None,
        reasoning_dynamic: bool | None = None,
        provider_options: dict[str, object] | None = None,
        trace_path: Path | str | None = None,
    ):
        resolved_model = model or self.DEFAULT_MODELS.get(
            provider, "gemini-3.1-flash-lite-preview"
        )
        super().__init__(
            project_id=project_id,
            model=resolved_model,
            location=location,
            temperature=temperature,
            provider=provider,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
            reasoning_dynamic=reasoning_dynamic,
            provider_options=provider_options,
            trace_path=trace_path,
        )
        self._segments = segments
        self._valid_names = {s["name"] for s in segments}

    def _get_system_instruction(self, num_lines: int) -> str:
        segments_text = ""
        for seg in self._segments:
            segments_text += f'- {seg["name"]}: {seg.get("description", "")}\n'
            if seg.get("cues"):
                segments_text += f'  Common cue phrases: {", ".join(seg["cues"])}\n'

        return (
            "You are analyzing a Japanese radio show or program transcript for segment transitions.\n"
            f"Your output must consist of exactly {num_lines} items.\n\n"
            "Task: Identify when the program transitions from one recurring segment (corner) to another.\n"
            "For each subtitle line, if it marks the START of a new segment, set 'corner' to that segment's name.\n"
            "Otherwise, set 'corner' to null.\n\n"
            f"Segments in this program:\n{segments_text}\n"
            "Instructions:\n"
            "1. Only mark the FIRST line of a new segment, not every line within it.\n"
            "2. Use neighboring lines for context. This is sequential dialogue from one episode.\n"
            "3. A transition can be signaled by an explicit cue phrase, a change in topic, or the host introducing a new segment.\n"
            "4. Do not mark a segment that is already in progress — only mark transitions.\n"
            "5. If unsure whether a transition occurred, prefer null (do not mark).\n"
            "6. Return valid JSON only.\n"
            "7. Return the exact same number of items as the input.\n"
            "8. Each item must contain exactly two fields: 'id' and 'corner'.\n"
            "9. 'corner' must be one of the segment names listed above, or null.\n"
        )

    def classify_window(
        self, lines: list[tuple[int, SubtitleLine]]
    ) -> dict[int, str | None]:
        if not lines:
            return {}

        system_instruction = self._get_system_instruction(len(lines))
        payload = [
            {
                "id": line_id,
                "text": line.text,
            }
            for line_id, line in lines
        ]
        contents = json.dumps(payload, ensure_ascii=False, indent=2)

        decisions, diagnostics = self._run_structured_output(
            user_prompt=contents,
            system_prompt=system_instruction,
            output_type=list[CornerDecision],
            operation_name="LLM corner classifier",
            output_name="corner_decisions",
        )

        try:
            ordered_decisions = sorted(decisions, key=lambda item: item.id)
            returned_ids = [item.id for item in ordered_decisions]

            if returned_ids != [line_id for line_id, _ in lines]:
                raise ValueError(f"returned ids were {returned_ids!r}")

            result: dict[int, str | None] = {}
            for item in ordered_decisions:
                corner = item.corner
                if corner and corner not in self._valid_names:
                    corner = None
                result[item.id] = corner
            return result
        except Exception as exc:
            raise VertexResponseShapeError(
                "LLM corner classifier returned JSON with an unexpected structure: "
                f"{exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc


def classify_corners_with_vertex(
    lines: list[SubtitleLine],
    segments: list[dict],
    config: dict,
) -> list[str | None]:
    if not lines:
        return []

    provider = config.get("provider", "google-vertex")
    project_id = config.get("project_id")
    if provider == "google-vertex" and not project_id:
        raise ValueError(
            "corners Vertex mode requires a Google Cloud project id."
        )

    classifier = VertexCornerClassifier(
        project_id=project_id,
        segments=segments,
        model=config.get("model"),
        location=config.get("location", "global"),
        provider=provider,
        reasoning_effort=config.get("reasoning_effort", ReasoningEffort.MEDIUM),
        reasoning_budget_tokens=config.get("reasoning_budget_tokens"),
        reasoning_dynamic=config.get("reasoning_dynamic"),
        provider_options=config.get("provider_options"),
        trace_path=config.get("llm_trace_path"),
    )

    windows = _build_windows(lines, config)

    # Collect votes across windows
    votes: dict[int, list[str | None]] = {i: [] for i in range(len(lines))}
    for window in windows:
        decisions = classifier.classify_window(window)
        for line_id, corner in decisions.items():
            if 0 <= line_id < len(lines):
                votes[line_id].append(corner)

    # Resolve: any non-null vote wins (corner transitions are sparse)
    resolved: list[str | None] = []
    for i in range(len(lines)):
        non_null = [v for v in votes[i] if v is not None]
        if non_null:
            counts = Counter(non_null)
            resolved.append(counts.most_common(1)[0][0])
        else:
            resolved.append(None)

    return resolved


def _build_windows(
    lines: list[SubtitleLine], config: dict
) -> list[list[tuple[int, SubtitleLine]]]:
    scope = str(config.get("scope", "full_script")).lower()
    if scope == "full_script":
        return [[(i, line) for i, line in enumerate(lines)]]

    window_size = max(int(config.get("window_size", 10)), 1)
    window_overlap = max(int(config.get("window_overlap", 3)), 0)

    step = max(window_size - window_overlap, 1)
    windows: list[list[tuple[int, SubtitleLine]]] = []
    start = 0
    while start < len(lines):
        end = min(start + window_size, len(lines))
        windows.append([(i, lines[i]) for i in range(start, end)])
        if end >= len(lines):
            break
        start += step
    return windows
