import json
import logging
from pathlib import Path
from typing import Sequence

from autosub.core.schemas import SubtitleLine, TranscriptionResult
from autosub.pipeline.format import chunker
from autosub.pipeline.format import generator
from autosub.pipeline.format.normalizer import (
    apply_normalization,
    apply_replacements_with_spans,
)
from autosub.pipeline.format.split_utils import find_split_time, partition_spans
from autosub.pipeline.format.timing import apply_timing_rules

logger = logging.getLogger(__name__)

__all__ = ["apply_replacements_with_spans", "apply_split_after", "format_subtitles"]

VALID_ENGINES: dict[str, set[str]] = {
    "radio_discourse": {"rules", "llm", "hybrid"},
    "corners": {"cues", "llm", "hybrid"},
}
TRAILING_SPLIT_PUNCTUATION = "。！？!?、,"


def _stage_trace_path(output_ass_path: Path, stage_name: str) -> Path:
    return output_ass_path.with_suffix(f".{stage_name}.llm_trace.jsonl")


def _stage_edit_audit_path(output_ass_path: Path, stage_name: str) -> Path:
    return output_ass_path.with_suffix(f".{stage_name}.edit_audit.tsv")


def _validate_engine(engine: str, extension: str, valid: set[str]) -> None:
    if engine not in valid:
        logger.warning(
            f"Unknown engine '{engine}' for {extension} extension. "
            f"Supported values: {', '.join(sorted(valid))}. "
            f"Falling back to deterministic-only mode."
        )


def _apply_combined_extensions(
    lines: list[SubtitleLine],
    radio_config: dict,
    corners_config: dict,
    output_ass_path: Path,
) -> list[SubtitleLine]:
    """Run radio_discourse + corners in a single LLM call."""
    from autosub.core.config import PROJECT_ID
    from autosub.core.errors import VertexError
    from autosub.extensions.combined_classifier import classify_combined
    from autosub.extensions.corners.main import dedup_consecutive, detect_by_cues
    from autosub.extensions.radio_discourse.main import (
        _normalize_greetings,
        classify_role,
        split_host_meta_suffix,
    )

    greetings = _normalize_greetings(radio_config.get("greetings", []))
    if greetings:
        lines = apply_split_after(lines, greetings, ensure_terminal_punctuation=True)

    processed: list[SubtitleLine] = []
    if radio_config.get("split_framing_phrases", True):
        for line in lines:
            processed.extend(split_host_meta_suffix(line))
    else:
        processed = list(lines)

    fallback_roles: list[str | None] = []
    previous_role: str | None = None
    for line in processed:
        role = classify_role(line.text, previous_role)
        fallback_roles.append(role)
        previous_role = role

    segments = corners_config.get("segments", [])
    cue_corners = detect_by_cues(processed, segments)

    # Build combined config: start from radio_config, then layer in corners
    # settings as fallbacks so corners-specific LLM config takes effect when
    # radio_discourse doesn't specify a given setting.
    combined_config = dict(radio_config)
    for key in (
        "model",
        "location",
        "provider",
        "reasoning_effort",
        "reasoning_budget_tokens",
        "reasoning_dynamic",
        "provider_options",
    ):
        if key not in combined_config and key in corners_config:
            combined_config[key] = corners_config[key]
    combined_config.setdefault("project_id", PROJECT_ID)
    llm_trace_path = _stage_trace_path(output_ass_path, "combined")
    combined_config.setdefault("llm_trace_path", llm_trace_path)
    llm_trace_path.unlink(missing_ok=True)

    try:
        roles, corners = classify_combined(
            processed, fallback_roles, segments, combined_config
        )
    except VertexError:
        radio_engine = str(radio_config.get("engine", "rules")).lower()
        corners_engine = str(corners_config.get("engine", "hybrid")).lower()
        if radio_engine == "llm" or corners_engine == "llm":
            raise
        logger.warning(
            "Combined classification failed; falling back to rules + cues.",
            exc_info=True,
        )
        roles = fallback_roles
        corners = cue_corners

    # Merge LLM corners with cue fallback
    merged_corners: list[str | None] = []
    for llm_c, cue_c in zip(corners, cue_corners, strict=False):
        merged_corners.append(llm_c if llm_c is not None else cue_c)
    merged_corners = dedup_consecutive(merged_corners)

    label_roles = radio_config.get("label_roles", True)
    result: list[SubtitleLine] = []
    for line, role, corner in zip(processed, roles, merged_corners, strict=False):
        result.append(
            SubtitleLine(
                text=line.text,
                start_time=line.start_time,
                end_time=line.end_time,
                speaker=line.speaker,
                role=role if label_roles else None,
                corner=corner,
            )
        )

    return result


def _segments_to_lines(transcript: TranscriptionResult) -> list[SubtitleLine]:
    lines: list[SubtitleLine] = []
    for segment in transcript.segments:
        lines.append(
            SubtitleLine(
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker=segment.speaker,
                words=list(segment.words),
            )
        )
    lines.sort(key=lambda line: line.start_time)
    return lines


def _initial_lines(transcript: TranscriptionResult) -> list[SubtitleLine]:
    backend = transcript.metadata.backend if transcript.metadata else None
    if backend == "whisperx" and transcript.segments:
        logger.info("Using transcript segments as initial subtitle lines.")
        return _segments_to_lines(transcript)
    return chunker.chunk_words_to_lines(transcript.words)


def _load_transcript(input_json_path: Path) -> TranscriptionResult:
    if not input_json_path.exists():
        raise FileNotFoundError(f"Transcript JSON file not found: {input_json_path}")

    logger.info("Loading transcript from %s...", input_json_path)
    with input_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return TranscriptionResult(**data)


def _initial_lines_from_inputs(
    input_json_paths: Path | Sequence[Path],
) -> list[SubtitleLine]:
    if isinstance(input_json_paths, Path):
        normalized_paths = [input_json_paths]
    else:
        normalized_paths = list(input_json_paths)

    if not normalized_paths:
        raise ValueError("At least one transcript JSON path is required.")

    merged_lines: list[SubtitleLine] = []
    for input_json_path in normalized_paths:
        transcript = _load_transcript(input_json_path)
        transcript_lines = _initial_lines(transcript)
        logger.info(
            "Generated %d initial subtitle lines from %s.",
            len(transcript_lines),
            input_json_path,
        )
        merged_lines.extend(transcript_lines)

    merged_lines.sort(key=lambda line: (line.start_time, line.end_time))
    return merged_lines


def _split_line_after(line: SubtitleLine, split_after: list[str]) -> list[SubtitleLine]:
    """Split a single line after every occurrence of any phrase in split_after."""
    return _split_line_after_with_options(
        line, split_after, ensure_terminal_punctuation=False
    )


def _split_line_after_with_options(
    line: SubtitleLine,
    split_after: list[str],
    *,
    ensure_terminal_punctuation: bool,
) -> list[SubtitleLine]:
    """Split a single line after every occurrence of any phrase in split_after."""
    split_positions: set[int] = set()
    for phrase in split_after:
        pos = 0
        while True:
            idx = line.text.find(phrase, pos)
            if idx == -1:
                break
            end_pos = idx + len(phrase)
            while (
                end_pos < len(line.text)
                and line.text[end_pos] in TRAILING_SPLIT_PUNCTUATION
            ):
                end_pos += 1
            if end_pos < len(line.text):
                split_positions.add(end_pos)
            pos = idx + 1

    if not split_positions:
        return [line]

    sorted_positions = sorted(split_positions)
    split_times = [find_split_time(line, pos) for pos in sorted_positions]

    text_boundaries = [0] + sorted_positions + [len(line.text)]
    time_boundaries = [line.start_time] + split_times + [line.end_time]

    result: list[SubtitleLine] = []
    current_spans = list(line.replacement_spans)

    for i in range(len(text_boundaries) - 1):
        txs = text_boundaries[i]
        txe = text_boundaries[i + 1]
        ts = time_boundaries[i]
        te = time_boundaries[i + 1]
        is_last = i == len(text_boundaries) - 2

        # Partition spans: the split position within the current span coordinate system
        # is always the length of this segment (txe - txs) because current_spans has
        # already been adjusted by prior iterations.
        if not is_last:
            seg_span_len = txe - txs
            seg_spans, current_spans = partition_spans(current_spans, seg_span_len)
        else:
            seg_spans = current_spans

        if is_last:
            seg_words = [w for w in line.words if w.end_time > ts]
        else:
            seg_words = [w for w in line.words if w.end_time > ts and w.end_time <= te]

        result.append(
            SubtitleLine(
                text=_normalize_split_text(
                    line.text[txs:txe],
                    ensure_terminal_punctuation=ensure_terminal_punctuation
                    and not is_last,
                ),
                start_time=ts,
                end_time=te,
                speaker=line.speaker,
                role=line.role,
                corner=line.corner,
                words=seg_words,
                replacement_spans=seg_spans,
            )
        )

    return result


def apply_split_after(
    lines: list[SubtitleLine],
    split_after: list[str],
    *,
    ensure_terminal_punctuation: bool = False,
) -> list[SubtitleLine]:
    """Split every line after each occurrence of any phrase in split_after."""
    result: list[SubtitleLine] = []
    for line in lines:
        result.extend(
            _split_line_after_with_options(
                line,
                split_after,
                ensure_terminal_punctuation=ensure_terminal_punctuation,
            )
        )
    return result


def _normalize_split_text(text: str, *, ensure_terminal_punctuation: bool) -> str:
    if not ensure_terminal_punctuation or not text:
        return text
    if text.endswith(tuple(TRAILING_SPLIT_PUNCTUATION)):
        return text
    return f"{text}。"


def format_subtitles(
    input_json_path: Path | Sequence[Path],
    output_ass_path: Path,
    keyframes: list[int] | None = None,
    video_duration_ms: int | None = None,
    timing_config: dict | None = None,
    extensions_config: dict | None = None,
    normalizer_config: dict | None = None,
    replacements: dict[str, str] | None = None,
) -> None:
    """
    Reads one or more transcript.json files, chunks the transcribed words into
    semantic lines, merges the initial line sets, applies timing rules (gap
    snapping, min duration, keyframes), and generates an output .ass subtitle file.
    """
    logger.info("Chunking transcript into semantic subtitle lines...")
    lines = _initial_lines_from_inputs(input_json_path)
    logger.info(f"Generated {len(lines)} subtitle lines.")

    if replacements and normalizer_config:
        raise ValueError(
            "format_subtitles received both replacements and normalizer_config; use only one."
        )
    if replacements:
        normalizer_config = {"engine": "exact", "replacements": replacements}
    if normalizer_config:
        normalizer_engine = str(normalizer_config.get("engine", "exact")).lower()
        if normalizer_engine == "llm":
            from autosub.core.config import PROJECT_ID

            llm_trace_path = _stage_trace_path(output_ass_path, "normalizer")
            edit_audit_path = _stage_edit_audit_path(output_ass_path, "normalizer")
            llm_trace_path.unlink(missing_ok=True)
            edit_audit_path.unlink(missing_ok=True)
            normalizer_config = dict(normalizer_config)
            normalizer_config.setdefault("project_id", PROJECT_ID)
            normalizer_config.setdefault("llm_trace_path", llm_trace_path)
            normalizer_config.setdefault("edit_audit_path", edit_audit_path)
        lines = apply_normalization(lines, normalizer_config)

    if not extensions_config:
        extensions_config = {}

    radio_discourse_config = extensions_config.get("radio_discourse", {})
    corners_config = extensions_config.get("corners", {})

    # Determine whether the combined path will run, so we can skip
    # the standalone radio_discourse call and avoid a wasted LLM pass.
    radio_enabled = radio_discourse_config.get("enabled", False)
    corners_enabled = corners_config.get("enabled", False)
    radio_engine = str(radio_discourse_config.get("engine", "rules")).lower()
    corners_engine = str(corners_config.get("engine", "hybrid")).lower()

    _validate_engine(radio_engine, "radio_discourse", VALID_ENGINES["radio_discourse"])
    _validate_engine(corners_engine, "corners", VALID_ENGINES["corners"])

    use_combined = (
        radio_enabled
        and corners_enabled
        and radio_engine in {"llm", "hybrid"}
        and corners_engine in {"llm", "hybrid"}
    )

    if radio_enabled and not use_combined:
        logger.info("Applying radio discourse extension...")
        from autosub.extensions.radio_discourse.main import apply_radio_discourse

        if radio_engine in {"llm", "hybrid"}:
            llm_trace_path = _stage_trace_path(output_ass_path, "radio_discourse")
            radio_discourse_config = dict(radio_discourse_config)
            radio_discourse_config.setdefault("llm_trace_path", llm_trace_path)
            llm_trace_path.unlink(missing_ok=True)

        lines = apply_radio_discourse(lines, radio_discourse_config)
        logger.info(f"Radio discourse extension produced {len(lines)} subtitle lines.")

    if corners_enabled:
        if use_combined:
            logger.info("Running combined radio discourse + corners classification...")
            lines = _apply_combined_extensions(
                lines, radio_discourse_config, corners_config, output_ass_path
            )
        else:
            # Standalone corners (cues-only, or radio_discourse not using LLM)
            logger.info("Applying corners extension...")
            from autosub.extensions.corners.main import apply_corners

            if corners_engine in {"llm", "hybrid"}:
                llm_trace_path = _stage_trace_path(output_ass_path, "corners")
                llm_trace_path.unlink(missing_ok=True)
                corners_config = dict(corners_config)
                corners_config.setdefault("llm_trace_path", llm_trace_path)

            lines = apply_corners(lines, corners_config)
            detected = sum(1 for line in lines if line.corner)
            logger.info(f"Corners extension detected {detected} transitions.")

    logger.info("Applying timing rules (snapping, keyframes, min duration)...")
    if not timing_config:
        timing_config = {}

    lines = apply_timing_rules(
        lines,
        keyframes_ms=keyframes,
        video_duration_ms=video_duration_ms,
        min_duration_ms=timing_config.get("min_duration_ms", 500),
        snap_threshold_ms=timing_config.get("snap_threshold_ms", 250),
        conditional_snap_threshold_ms=timing_config.get(
            "conditional_snap_threshold_ms", 500
        ),
    )

    logger.info(f"Writing .ass file to {output_ass_path}...")
    generator.generate_ass_file(lines, output_ass_path)
    trace_paths = [
        _stage_trace_path(output_ass_path, name)
        for name in ("normalizer", "radio_discourse", "corners", "combined")
    ]
    for trace_path in trace_paths:
        if trace_path.exists():
            logger.info(f"Wrote LLM trace to {trace_path}.")
    logger.info("Subtitle formatting complete!")
