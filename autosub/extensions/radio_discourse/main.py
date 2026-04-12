from __future__ import annotations

import logging
import re

from autosub.core.config import PROJECT_ID
from autosub.core.errors import VertexError
from autosub.core.schemas import SubtitleLine
from autosub.extensions.radio_discourse.classifier import classify_roles_with_vertex

logger = logging.getLogger(__name__)

HOST_META_SUFFIXES = (
    "といただきました。",
    "といただきました",
    "とのことです。",
    "とのことです",
    "というお便りでした。",
    "というお便りでした",
    "と書いてありました。",
    "と書いてありました",
)
HOST_META_EXACT = {
    "といただきました。",
    "といただきました",
    "とのことです。",
    "とのことです",
    "というお便りでした。",
    "というお便りでした",
}
HOST_REACTION_EXACT = {
    "おお",
    "おお。",
    "おお!",
    "おお！",
    "ほう",
    "ほう。",
    "へえ",
    "へえ。",
    "すごい",
    "すごい。",
    "すごいな",
    "すごいな。",
    "ありがとう",
    "ありがとう。",
    "ありがとうね",
    "ありがとうね。",
    "嬉しい",
    "嬉しい。",
}
HOST_REACTION_PATTERNS = (
    re.compile(r"^(おお|へえ|ほう|なるほど)[。！!\?？]*$"),
    re.compile(r"^(すごい|嬉しい|ありがとう)(ね)?[。！!\?？]*$"),
)
LISTENER_MAIL_PATTERNS = (
    re.compile(
        r".*(です|ます|でした|ました|と思います|いただきました|どうですか)[。！!\?？]?$"
    ),
    re.compile(r".*(僕|私|わたし|自分)は.*"),
)
TRAILING_PUNCTUATION = "。！？!?"


def apply_radio_discourse(
    lines: list[SubtitleLine], config: dict | None = None
) -> list[SubtitleLine]:
    if not lines:
        return []

    if config is None:
        config = {}

    processed_lines: list[SubtitleLine] = []
    for line in lines:
        if config.get("split_framing_phrases", True):
            processed_lines.extend(split_host_meta_suffix(line))
        else:
            processed_lines.append(line)

    fallback_roles: list[str | None] = []
    previous_role: str | None = None
    for line in processed_lines:
        role = classify_role(line.text, previous_role)
        fallback_roles.append(role)
        previous_role = role

    engine = str(config.get("engine", "rules")).lower()
    resolved_roles = fallback_roles
    if engine in {"llm", "hybrid"}:
        llm_config = dict(config)
        llm_config.setdefault("project_id", PROJECT_ID)
        try:
            resolved_roles = classify_roles_with_vertex(
                processed_lines,
                fallback_roles,
                llm_config,
            )
        except VertexError:
            if engine == "llm":
                raise
            logger.warning(
                "Vertex radio discourse classification failed; falling back to rules.",
                exc_info=True,
            )

    label_roles = config.get("label_roles", True)
    classified_lines: list[SubtitleLine] = []
    for line, role in zip(processed_lines, resolved_roles, strict=False):
        classified_lines.append(
            SubtitleLine(
                text=line.text,
                start_time=line.start_time,
                end_time=line.end_time,
                speaker=line.speaker,
                role=role if label_roles else None,
            )
        )

    return classified_lines


def split_host_meta_suffix(line: SubtitleLine) -> list[SubtitleLine]:
    text = line.text.strip()
    for suffix in HOST_META_SUFFIXES:
        if not text.endswith(suffix):
            continue

        main_text = text[: -len(suffix)].rstrip()
        if not main_text:
            return [line]

        main_text = _ensure_terminal_punctuation(main_text)
        total_duration = max(line.end_time - line.start_time, 0.0)
        if total_duration <= 0:
            return [
                SubtitleLine(
                    text=main_text,
                    start_time=line.start_time,
                    end_time=line.end_time,
                    speaker=line.speaker,
                ),
                SubtitleLine(
                    text=suffix,
                    start_time=line.end_time,
                    end_time=line.end_time,
                    speaker=line.speaker,
                ),
            ]

        boundary_ratio = len(main_text) / max(len(main_text) + len(suffix), 1)
        split_time = line.start_time + total_duration * boundary_ratio

        return [
            SubtitleLine(
                text=main_text,
                start_time=line.start_time,
                end_time=split_time,
                speaker=line.speaker,
            ),
            SubtitleLine(
                text=suffix,
                start_time=split_time,
                end_time=line.end_time,
                speaker=line.speaker,
            ),
        ]

    return [line]


def classify_role(text: str, previous_role: str | None) -> str | None:
    stripped = text.strip()
    if not stripped:
        return previous_role

    if _is_host_meta(stripped):
        return "host_meta"

    if _is_host_reaction(stripped):
        return "host"

    if _looks_like_listener_mail(stripped):
        return "listener_mail"

    if len(stripped) <= 12:
        return "host"

    if previous_role == "listener_mail" and len(stripped) >= 12:
        return "listener_mail"

    return previous_role


def _is_host_meta(text: str) -> bool:
    return text in HOST_META_EXACT


def _is_host_reaction(text: str) -> bool:
    if text in HOST_REACTION_EXACT:
        return True
    return any(pattern.match(text) for pattern in HOST_REACTION_PATTERNS)


def _looks_like_listener_mail(text: str) -> bool:
    return any(pattern.match(text) for pattern in LISTENER_MAIL_PATTERNS)


def _ensure_terminal_punctuation(text: str) -> str:
    if text.endswith(tuple(TRAILING_PUNCTUATION)):
        return text
    return f"{text}。"
