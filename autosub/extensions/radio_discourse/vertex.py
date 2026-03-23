from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Iterable, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)

ROLE_VALUES = ("host", "listener_mail", "host_meta")


class RadioDiscourseDecision(BaseModel):
    id: int
    role: Literal["host", "listener_mail", "host_meta"]


class VertexRadioDiscourseClassifier:
    def __init__(
        self,
        project_id: str,
        model: str = "gemini-3-flash-preview",
        location: str = "global",
    ):
        self.project_id = project_id
        self.model = model
        self.location = location

    def _build_prompt(self, lines: list[tuple[int, SubtitleLine]]) -> str:
        payload = [
            {
                "id": line_id,
                "text": line.text,
            }
            for line_id, line in lines
        ]
        payload_str = json.dumps(payload, ensure_ascii=False, indent=2)

        return (
            "You are analyzing a Japanese solo voice-actress radio show transcript for subtitle segmentation.\n"
            "Task: Classify each subtitle line as one of three discourse roles.\n\n"
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
            "13. Each item must contain exactly two fields: 'id' and 'role'.\n\n"
            f"Input JSON:\n{payload_str}"
        )

    def classify_window(self, lines: list[tuple[int, SubtitleLine]]) -> dict[int, str]:
        if not lines:
            return {}

        client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
        )
        prompt = self._build_prompt(lines)

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[RadioDiscourseDecision],
                temperature=0.1,
            ),
        )

        if not response.text:
            raise ValueError(
                "Vertex radio discourse classifier returned an empty response."
            )

        response_json = json.loads(response.text)
        response_json.sort(key=lambda item: item["id"])
        return {item["id"]: item["role"] for item in response_json}


def classify_roles_with_vertex(
    lines: list[SubtitleLine],
    fallback_roles: list[str | None],
    config: dict,
) -> list[str | None]:
    if not lines:
        return []

    project_id = config.get("project_id")
    if not project_id:
        raise ValueError(
            "radio_discourse Vertex mode requires a Google Cloud project id."
        )

    classifier = VertexRadioDiscourseClassifier(
        project_id=project_id,
        model=config.get("model", "gemini-3-flash-preview"),
        location=config.get("location", "global"),
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
