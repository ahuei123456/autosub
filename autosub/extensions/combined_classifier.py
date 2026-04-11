"""Combined radio_discourse + corners classifier.

When both extensions use LLM/hybrid engine, this module runs a single LLM call
that classifies discourse roles AND detects corner transitions simultaneously.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from autosub.core.errors import VertexResponseShapeError
from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.core.schemas import SubtitleLine
from autosub.extensions.radio_discourse.classifier import ROLE_VALUES

logger = logging.getLogger(__name__)


class CombinedDecision(BaseModel):
    id: int
    role: Literal["host", "listener_mail", "host_meta"]
    corner: str | None = None


class CombinedClassifier(BaseStructuredLLM):
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
        self._valid_corner_names = {s["name"] for s in segments}

    def _get_system_instruction(self, num_lines: int) -> str:
        segments_text = ""
        for seg in self._segments:
            segments_text += f'- {seg["name"]}: {seg.get("description", "")}\n'
            if seg.get("cues"):
                segments_text += f'  Common cue phrases: {", ".join(seg["cues"])}\n'

        return (
            "You are analyzing a Japanese solo voice-actress radio show transcript.\n"
            f"Your output must consist of exactly {num_lines} items.\n\n"
            "You have TWO tasks for each subtitle line:\n\n"
            "TASK 1 — Discourse Role Classification:\n"
            "Classify each line as one of three discourse roles.\n"
            "Role definitions:\n"
            "1. host: the host's own live speech, reactions, commentary, ad-libbing, or monologue.\n"
            "2. listener_mail: text from a listener message, question, or submission that the host is reading aloud.\n"
            "3. host_meta: the host's framing around listener mail, such as introducing a message, naming the sender, or quotative wrap-up like といただきました.\n\n"
            "Role classification rules:\n"
            "1. Use neighboring lines for context. This is sequential dialogue from one radio episode.\n"
            "2. Do not rewrite or normalize the text. Only classify each input line.\n"
            "3. A listener-mail block can span multiple consecutive lines.\n"
            "4. Short host reactions like おお。ありがとう。なるほど。 are usually host.\n"
            "5. Lines that introduce or close out a listener message are usually host_meta, not listener_mail.\n"
            "6. If a line contains only a quotative wrap-up such as といただきました, label it host_meta.\n"
            "7. Greeting edge case: if a listener greeting is immediately echoed or answered by the host in the next line, the echo/answer is host, not listener_mail.\n\n"
            "TASK 2 — Segment (Corner) Detection:\n"
            "Identify when the program transitions from one recurring segment to another.\n"
            "If a line marks the START of a new segment, set 'corner' to that segment's name.\n"
            "Otherwise, set 'corner' to null.\n\n"
            f"Segments in this program:\n{segments_text}\n"
            "Corner detection rules:\n"
            "1. Only mark the FIRST line of a new segment, not every line within it.\n"
            "2. A transition can be signaled by an explicit cue phrase, a change in topic, or the host introducing a new segment.\n"
            "3. Do not mark a segment that is already in progress.\n"
            "4. If unsure whether a transition occurred, prefer null.\n"
            "5. 'corner' must be one of the segment names listed above, or null.\n\n"
            "Output format:\n"
            "1. Return valid JSON only.\n"
            "2. Return the exact same number of items as the input.\n"
            "3. Each item must contain exactly three fields: 'id', 'role', and 'corner'.\n"
        )

    def classify_window(
        self, lines: list[tuple[int, SubtitleLine]]
    ) -> dict[int, tuple[str, str | None]]:
        """Classify a window of lines, returning (role, corner) per line."""
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
            output_type=list[CombinedDecision],
            operation_name="LLM combined classifier",
            output_name="combined_decisions",
        )

        try:
            ordered = sorted(decisions, key=lambda item: item.id)
            returned_ids = [item.id for item in ordered]

            if returned_ids != [line_id for line_id, _ in lines]:
                raise ValueError(f"returned ids were {returned_ids!r}")

            result: dict[int, tuple[str, str | None]] = {}
            for item in ordered:
                corner = item.corner
                if corner and corner not in self._valid_corner_names:
                    corner = None
                result[item.id] = (item.role, corner)
            return result
        except Exception as exc:
            raise VertexResponseShapeError(
                "LLM combined classifier returned JSON with an unexpected structure: "
                f"{exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc


def classify_combined(
    lines: list[SubtitleLine],
    fallback_roles: list[str | None],
    segments: list[dict],
    config: dict,
) -> tuple[list[str | None], list[str | None]]:
    """Run combined classification, returning (roles, corners) lists."""
    if not lines:
        return [], []

    provider = config.get("provider", "google-vertex")
    project_id = config.get("project_id")
    if provider == "google-vertex" and not project_id:
        raise ValueError(
            "Combined classifier Vertex mode requires a Google Cloud project id."
        )

    classifier = CombinedClassifier(
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

    role_votes: dict[int, list[str]] = {i: [] for i in range(len(lines))}
    corner_votes: dict[int, list[str | None]] = {i: [] for i in range(len(lines))}

    for window in windows:
        decisions = classifier.classify_window(window)
        for line_id, (role, corner) in decisions.items():
            if 0 <= line_id < len(lines):
                if role in ROLE_VALUES:
                    role_votes[line_id].append(role)
                corner_votes[line_id].append(corner)

    # Resolve roles (same logic as radio_discourse)
    resolved_roles: list[str | None] = []
    for i in range(len(lines)):
        resolved_roles.append(_resolve_role(role_votes[i], fallback_roles[i]))

    # Resolve corners (any non-null vote wins)
    resolved_corners: list[str | None] = []
    for i in range(len(lines)):
        non_null = [v for v in corner_votes[i] if v is not None]
        if non_null:
            counts = Counter(non_null)
            resolved_corners.append(counts.most_common(1)[0][0])
        else:
            resolved_corners.append(None)

    return resolved_roles, resolved_corners


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


def _resolve_role(votes: list[str], fallback_role: str | None) -> str | None:
    if not votes:
        return fallback_role

    counts = Counter(votes)
    top_count = max(counts.values())
    top_roles = {role for role, count in counts.items() if count == top_count}

    if fallback_role in top_roles:
        return fallback_role

    for role in ("host_meta", "listener_mail", "host"):
        if role in top_roles:
            return role

    return fallback_role
