from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel

from autosub.core.errors import VertexResponseShapeError
from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)

ROLE_VALUES = ("host", "listener_mail", "host_meta")


class RadioDiscourseDecision(BaseModel):
    id: int
    role: Literal["host", "listener_mail", "host_meta"]


class VertexRadioDiscourseClassifier(BaseStructuredLLM):
    DEFAULT_MODELS = {
        "google-vertex": "gemini-3.1-flash-lite-preview",
        "anthropic-vertex": "claude-haiku-4-5",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
    }

    def __init__(
        self,
        *,
        project_id: str | None,
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

    def _get_system_instruction(self, num_lines: int) -> str:
        return (
            "You are analyzing a Japanese solo voice-actress radio show transcript for subtitle segmentation.\n"
            "Task: Classify each subtitle line as one of three discourse roles.\n"
            f"Your output must consist of exactly {num_lines} items.\n\n"
            "Role definitions:\n"
            "1. host: the host's own live speech, reactions, commentary, ad-libbing, or monologue.\n"
            "2. listener_mail: text from a listener message, question, or submission that the host is reading aloud.\n"
            "3. host_meta: the host's framing around listener mail, such as introducing a message, naming the sender, or quotative wrap-up like といただきました.\n\n"
            "Instructions:\n"
            "1. Use neighboring lines for context. This is sequential dialogue from one radio episode.\n"
            "2. Do not rewrite or normalize the text. Only classify each input line.\n"
            "3. A listener-mail block can span multiple consecutive lines.\n"
            "4. Short host reactions like おお。ありがとう。なるほど。 are usually host.\n"
            "5. Lines that introduce or close out a listener message are usually host_meta, not listener_mail.\n"
            "6. If a line contains only a quotative wrap-up such as といただきました, label it host_meta.\n"
            "7. Greeting edge case: if a listener greeting is immediately echoed or answered by the host in the next line, the echo/answer is host, not listener_mail, even if the wording is similar.\n"
            "8. Example: line A 'のちゃん、の番は?' is listener_mail, but the next line 'の番は?' is host because it is the host echoing the greeting.\n"
            "9. Example: line A 'のんちゃんのこんばんは。' is listener_mail, but the next line 'ノンバーワン。' is host because it is the host responding to the greeting.\n"
            "10. If one line appears to mix a listener greeting with a host echo and then continues the actual letter, classify it according to the dominant informational content of the line. In most such cases, prefer listener_mail if the line mainly contains the listener's actual message content.\n"
            "11. Return valid JSON only.\n"
            "12. Return the exact same number of items as the input.\n"
            "13. Each item must contain exactly two fields: 'id' and 'role'.\n"
        )

    def classify_window(self, lines: list[tuple[int, SubtitleLine]]) -> dict[int, str]:
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
            output_type=list[RadioDiscourseDecision],
            operation_name="LLM radio discourse classifier",
            output_name="radio_discourse_roles",
        )

        try:
            ordered_decisions = sorted(decisions, key=lambda item: item.id)
            returned_ids = [item.id for item in ordered_decisions]

            if returned_ids != [line_id for line_id, _ in lines]:
                raise ValueError(f"returned ids were {returned_ids!r}")

            return {item.id: item.role for item in ordered_decisions}
        except Exception as exc:
            raise VertexResponseShapeError(
                "LLM radio discourse classifier returned JSON with an unexpected structure: "
                f"{exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc


def classify_roles_with_vertex(
    lines: list[SubtitleLine],
    fallback_roles: list[str | None],
    config: dict,
) -> list[str | None]:
    if not lines:
        return []

    provider = config.get("provider", "google-vertex")
    project_id = config.get("project_id")
    if provider in {"google-vertex", "anthropic-vertex"} and not project_id:
        raise ValueError(
            "radio_discourse Vertex mode requires a Google Cloud project id."
        )

    classifier = VertexRadioDiscourseClassifier(
        project_id=project_id,
        model=config.get("model"),
        location=config.get("location", "global"),
        provider=provider,
        reasoning_effort=config.get("reasoning_effort", ReasoningEffort.MEDIUM),
        reasoning_budget_tokens=config.get("reasoning_budget_tokens"),
        reasoning_dynamic=config.get("reasoning_dynamic"),
        provider_options=config.get("provider_options"),
        trace_path=config.get("llm_trace_path"),
    )

    windows = _build_windows_for_config(lines, config)

    votes: dict[int, list[str]] = {index: [] for index in range(len(lines))}
    for window in windows:
        decisions = classifier.classify_window(window)
        for line_id, role in decisions.items():
            if 0 <= line_id < len(lines) and role in ROLE_VALUES:
                votes[line_id].append(role)

    resolved_roles: list[str | None] = []
    for index, vote_list in votes.items():
        resolved_roles.append(_resolve_role(vote_list, fallback_roles[index]))

    return resolved_roles


def _build_windows_for_config(
    lines: list[SubtitleLine], config: dict
) -> list[list[tuple[int, SubtitleLine]]]:
    scope = str(config.get("scope", "full_script")).lower()
    if scope == "full_script":
        return [[(index, line) for index, line in enumerate(lines)]]

    window_size = max(int(config.get("window_size", 10)), 1)
    window_overlap = max(int(config.get("window_overlap", 3)), 0)
    return _build_windows(lines, window_size, window_overlap)


def _build_windows(
    lines: list[SubtitleLine], window_size: int, window_overlap: int
) -> list[list[tuple[int, SubtitleLine]]]:
    if not lines:
        return []

    step = max(window_size - window_overlap, 1)
    windows: list[list[tuple[int, SubtitleLine]]] = []
    start = 0
    while start < len(lines):
        end = min(start + window_size, len(lines))
        windows.append([(index, lines[index]) for index in range(start, end)])
        if end >= len(lines):
            break
        start += step
    return windows


def _resolve_role(votes: Iterable[str], fallback_role: str | None) -> str | None:
    vote_list = list(votes)
    if not vote_list:
        return fallback_role

    counts = Counter(vote_list)
    top_count = max(counts.values())
    top_roles = {role for role, count in counts.items() if count == top_count}

    if fallback_role in top_roles:
        return fallback_role

    for role in ("host_meta", "listener_mail", "host"):
        if role in top_roles:
            return role

    return fallback_role
