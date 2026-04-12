from __future__ import annotations

import logging

from autosub.core.config import PROJECT_ID
from autosub.core.errors import VertexError
from autosub.core.schemas import SubtitleLine

logger = logging.getLogger(__name__)


def apply_corners(
    lines: list[SubtitleLine], config: dict | None = None
) -> list[SubtitleLine]:
    """Detect corner/segment transitions and mark the first line of each new segment.

    Supports these engines:
    - "cues": deterministic cue-phrase matching only
    - "llm": LLM-based detection (raises on failure)
    - "hybrid" (default): LLM with cue-based fallback on failure

    """
    if not lines:
        return []

    if config is None:
        config = {}

    segments = config.get("segments", [])
    if not segments:
        logger.warning("Corners extension enabled but no segments defined.")
        return lines

    # Log segment configuration
    for seg in segments:
        cues = seg.get("cues", [])
        logger.info(
            f"  Segment '{seg['name']}': {len(cues)} cues"
            + (f" [{', '.join(cues[:3])}{'...' if len(cues) > 3 else ''}]" if cues else "")
        )

    # Deterministic cue-based detection
    cue_corners = detect_by_cues(lines, segments)

    engine = str(config.get("engine", "hybrid")).lower()
    if engine in {"llm", "hybrid"}:
        llm_config = dict(config)
        llm_config.setdefault("project_id", PROJECT_ID)
        try:
            from autosub.extensions.corners.classifier import (
                classify_corners_with_vertex,
            )

            llm_corners = classify_corners_with_vertex(lines, segments, llm_config)
            llm_detected = [(i, c) for i, c in enumerate(llm_corners) if c]
            if llm_detected:
                logger.info(f"  LLM detected {len(llm_detected)} transition(s):")
                for i, name in llm_detected:
                    logger.info(f"    Line {i}: {name}")
            # Merge: LLM takes precedence, cue fills gaps
            resolved_corners = _merge_detections(cue_corners, llm_corners)
        except VertexError:
            if engine == "llm":
                raise
            logger.warning(
                "Vertex corner classification failed; falling back to cues.",
                exc_info=True,
            )
            resolved_corners = cue_corners
    else:
        resolved_corners = cue_corners

    # Deduplicate consecutive same-corner detections
    resolved_corners = dedup_consecutive(resolved_corners)

    result: list[SubtitleLine] = []
    for line, corner in zip(lines, resolved_corners, strict=False):
        result.append(
            SubtitleLine(
                text=line.text,
                start_time=line.start_time,
                end_time=line.end_time,
                speaker=line.speaker,
                role=line.role,
                corner=corner,
            )
        )

    detected = [c for c in resolved_corners if c is not None]
    if detected:
        logger.info(f"Corners detected: {', '.join(detected)}")

    return result


def detect_by_cues(
    lines: list[SubtitleLine], segments: list[dict]
) -> list[str | None]:
    """Scan lines for cue phrases and return corner names at transition points."""
    results: list[str | None] = [None] * len(lines)

    # Build cue → segment name mapping
    cue_map: list[tuple[str, str]] = []
    for seg in segments:
        for cue in seg.get("cues", []):
            cue_map.append((cue, seg["name"]))

    if not cue_map:
        return results

    for i, line in enumerate(lines):
        for cue, name in cue_map:
            if cue in line.text:
                results[i] = name
                logger.debug(f"  Cue match at line {i}: '{cue}' → {name}")
                break

    matched_names = {r for r in results if r is not None}
    all_names = {seg["name"] for seg in segments}
    unmatched = all_names - matched_names
    if unmatched:
        logger.info(f"  No cue matches for: {', '.join(sorted(unmatched))}")

    return results


def _merge_detections(
    cue_corners: list[str | None], llm_corners: list[str | None]
) -> list[str | None]:
    """Merge LLM and cue-based detections. LLM takes precedence."""
    merged: list[str | None] = []
    for cue, llm in zip(cue_corners, llm_corners, strict=False):
        merged.append(llm if llm is not None else cue)
    return merged


def dedup_consecutive(corners: list[str | None]) -> list[str | None]:
    """Remove consecutive duplicate corner names, keeping only the first.

    Only suppresses truly consecutive repeats. A None gap between two
    occurrences of the same corner resets tracking, so the second occurrence
    is preserved (e.g. a show that returns to "Fan Letter" after a "Song").
    """
    result: list[str | None] = []
    last_corner: str | None = None
    for corner in corners:
        if corner is not None and corner == last_corner:
            result.append(None)
        else:
            result.append(corner)
            last_corner = corner
    return result
