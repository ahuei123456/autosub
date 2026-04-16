from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel

from autosub.core.errors import VertexResponseShapeError
from autosub.core.llm import BaseStructuredLLM, ReasoningEffort
from autosub.core.schemas import ReplacementSpan, SubtitleLine

logger = logging.getLogger(__name__)
IGNORABLE_CONTEXT_PUNCTUATION = frozenset("、。！？!? \t　")


class NormalizerTerm(BaseModel):
    value: str
    explanation: str | None = None


class NormalizationEdit(BaseModel):
    line_id: int
    source_text: str
    replacement_text: str
    start_char: int
    end_char: int


class _ValidatedEdit(NamedTuple):
    source_text: str
    replacement_text: str
    start_char: int
    end_char: int


class _ValidationResult(NamedTuple):
    grouped_edits: dict[int, list[_ValidatedEdit]]
    errors: list[str]
    rejected_edits: list[NormalizationEdit]


class _ContextValidationResult(NamedTuple):
    accepted_edits: list[_ValidatedEdit]
    errors: list[str]
    rejected_edits: list[NormalizationEdit]


class _RangeOverrideResult(NamedTuple):
    resolved_edits: list[NormalizationEdit]
    unresolved_edits: list[NormalizationEdit]
    errors: list[str]


class NormalizerValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def _format_validation_errors(stage: str, errors: list[str]) -> str:
    lines = [f"{stage} validation errors ({len(errors)}):"]
    lines.extend(f"- {item}" for item in errors)
    return "\n".join(lines)


def _log_validation_errors(level: int, stage: str, errors: list[str]) -> None:
    if not errors:
        return
    logger.log(level, _format_validation_errors(stage, errors))


class LLMKeywordNormalizer(BaseStructuredLLM):
    DEFAULT_MODELS = {
        "google-vertex": "gemini-3.1-flash-lite-preview",
        "anthropic-vertex": "claude-haiku-4-5",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
        "openrouter": "google/gemini-3.1-flash-lite-preview",
    }

    def __init__(
        self,
        *,
        project_id: str | None,
        model: str | None = None,
        location: str = "global",
        temperature: float = 0.0,
        provider: str = "google-vertex",
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.MINIMAL,
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

    def _get_system_instruction(self) -> str:
        return (
            "You are normalizing Japanese ASR transcript text to approved canonical terms.\n"
            "Task: find exact contiguous substrings in the provided subtitle lines that should be replaced with one of the approved terms.\n\n"
            "Rules:\n"
            "1. Only return replacements for exact contiguous substrings that already exist in the line text.\n"
            "2. replacement_text may be either an approved term or a substring that helps the surrounding line resolve to an approved term.\n"
            "3. Do not rewrite surrounding words, punctuation, spacing, or grammar.\n"
            "4. Do not merge edits, split lines, or invent new text outside the replacement span.\n"
            "5. If a line is already correct or you are unsure, return no edit for that line.\n"
            "6. Do not return overlapping edits.\n"
            "7. start_char and end_char must index the original line text, with end_char exclusive.\n"
            "8. source_text must exactly match line.text[start_char:end_char].\n"
            "9. Return valid JSON only.\n"
        )

    @staticmethod
    def _build_payload(
        lines: list[SubtitleLine],
        terms: list[NormalizerTerm],
        *,
        accepted_edits: list[NormalizationEdit] | None = None,
        previous_edits: list[NormalizationEdit] | None = None,
        validation_errors: list[str] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "approved_terms": [
                term.model_dump(mode="json", exclude_none=True) for term in terms
            ],
            "lines": [
                {"id": index, "text": line.text} for index, line in enumerate(lines)
            ],
        }
        if previous_edits is not None:
            payload["previous_attempt"] = [
                edit.model_dump(mode="json") for edit in previous_edits
            ]
        if accepted_edits is not None:
            payload["accepted_edits"] = [
                edit.model_dump(mode="json") for edit in accepted_edits
            ]
        if validation_errors is not None:
            payload["validation_errors"] = validation_errors
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def propose_edits(
        self,
        lines: list[SubtitleLine],
        terms: list[NormalizerTerm],
    ) -> list[NormalizationEdit]:
        if not lines or not terms:
            return []

        user_prompt = self._build_payload(lines, terms)
        system_prompt = self._get_system_instruction()

        edits, diagnostics = self._run_structured_output(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            output_type=list[NormalizationEdit],
            operation_name="LLM keyword normalizer",
            output_name="normalization_edits",
        )

        try:
            return list(edits)
        except Exception as exc:
            raise VertexResponseShapeError(
                f"LLM keyword normalizer returned JSON with an unexpected structure: {exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc

    def propose_corrected_edits(
        self,
        lines: list[SubtitleLine],
        terms: list[NormalizerTerm],
        *,
        accepted_edits: list[NormalizationEdit],
        previous_edits: list[NormalizationEdit],
        validation_errors: list[str],
    ) -> list[NormalizationEdit]:
        if not lines or not terms:
            return []

        user_prompt = self._build_payload(
            lines,
            terms,
            accepted_edits=accepted_edits,
            previous_edits=previous_edits,
            validation_errors=validation_errors,
        )
        system_prompt = (
            self._get_system_instruction()
            + "\nCorrection pass instructions:\n"
            + "10. The previous_attempt was rejected for the listed validation_errors.\n"
            + "11. accepted_edits already passed validation. Do not repeat or modify them.\n"
            + "12. Return corrected edits only for the rejected previous_attempt items.\n"
            + "13. source_text, start_char, and end_char must still refer to the original unedited line text from lines, never to text after applying any previous edit.\n"
            + "14. Fix every listed validation error before returning.\n"
        )

        edits, diagnostics = self._run_structured_output(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            output_type=list[NormalizationEdit],
            operation_name="LLM keyword normalizer correction",
            output_name="normalization_edits",
        )

        try:
            return list(edits)
        except Exception as exc:
            raise VertexResponseShapeError(
                f"LLM keyword normalizer correction returned JSON with an unexpected structure: {exc}",
                diagnostics=diagnostics,
                project_id=self.project_id,
                model=self.model,
                location=self.location,
            ) from exc


def _apply_line_edits_with_spans(
    text: str,
    edits: list[_ValidatedEdit],
) -> tuple[str, list[ReplacementSpan]]:
    if not edits:
        return text, []

    edits = sorted(edits, key=lambda item: (item.start_char, item.end_char))
    result_parts: list[str] = []
    spans: list[ReplacementSpan] = []
    orig_pos = 0
    replaced_pos = 0

    for edit in edits:
        if edit.start_char < orig_pos:
            raise ValueError("Overlapping edits are not allowed.")
        if text[edit.start_char : edit.end_char] != edit.source_text:
            raise ValueError("Edit source text does not match the provided span.")

        if edit.start_char > orig_pos:
            chunk = text[orig_pos : edit.start_char]
            result_parts.append(chunk)
            replaced_pos += len(chunk)

        result_parts.append(edit.replacement_text)
        spans.append(
            ReplacementSpan(
                orig_start=edit.start_char,
                orig_end=edit.end_char,
                replaced_start=replaced_pos,
                replaced_end=replaced_pos + len(edit.replacement_text),
            )
        )
        replaced_pos += len(edit.replacement_text)
        orig_pos = edit.end_char

    if orig_pos < len(text):
        result_parts.append(text[orig_pos:])

    return "".join(result_parts), spans


def _apply_replacements_with_spans(
    text: str,
    replacements: dict[str, str],
) -> tuple[str, list[ReplacementSpan]]:
    """
    Apply all replacements to text in a single pass, returning the replaced text
    and a list of ReplacementSpan objects tracking each substitution.

    Replacements are applied longest-source-first to resolve ambiguity when
    multiple patterns could match at the same position. Overlapping matches are
    skipped (first match wins).
    """
    if not replacements:
        return text, []

    all_matches: list[tuple[int, int, str]] = []
    for old_str in replacements:
        pos = 0
        while True:
            idx = text.find(old_str, pos)
            if idx == -1:
                break
            all_matches.append((idx, idx + len(old_str), old_str))
            pos = idx + 1

    if not all_matches:
        return text, []

    all_matches.sort(key=lambda match: (match[0], -(match[1] - match[0])))

    accepted: list[tuple[int, int, str]] = []
    last_end = 0
    for start, end, old_str in all_matches:
        if start >= last_end:
            accepted.append((start, end, old_str))
            last_end = end

    validated_edits = [
        _ValidatedEdit(
            source_text=old_str,
            replacement_text=replacements[old_str],
            start_char=orig_start,
            end_char=orig_end,
        )
        for orig_start, orig_end, old_str in accepted
        if replacements[old_str] != old_str
    ]
    return _apply_line_edits_with_spans(text, validated_edits)


def apply_exact_normalization(
    lines: list[SubtitleLine],
    replacements: dict[str, str],
) -> list[SubtitleLine]:
    if not replacements:
        return lines

    logger.info("Applying %s exact text replacements...", len(replacements))
    for line in lines:
        new_text, spans = _apply_replacements_with_spans(line.text, replacements)
        line.text = new_text
        line.replacement_spans = spans
    return lines


def _collect_llm_edit_validation(
    lines: list[SubtitleLine],
    edits: list[NormalizationEdit],
    *,
    allowed_terms: set[str],
    existing_grouped_edits: dict[int, list[_ValidatedEdit]] | None = None,
) -> _ValidationResult:
    if existing_grouped_edits is None:
        existing_grouped_edits = {}
    provisional: dict[int, list[tuple[NormalizationEdit, _ValidatedEdit]]] = {}
    grouped: dict[int, list[_ValidatedEdit]] = {}
    errors: list[str] = []
    rejected_edits: list[NormalizationEdit] = []
    for edit in edits:
        if not 0 <= edit.line_id < len(lines):
            errors.append(
                f"Invalid line_id {edit.line_id} for edit source={edit.source_text!r} replacement={edit.replacement_text!r} range=[{edit.start_char}, {edit.end_char})."
            )
            rejected_edits.append(edit)
            continue
        if edit.start_char < 0 or edit.end_char <= edit.start_char:
            errors.append(
                f"Invalid edit range [{edit.start_char}, {edit.end_char}) for line {edit.line_id}. "
                f"line={lines[edit.line_id].text!r} source={edit.source_text!r} replacement={edit.replacement_text!r}."
            )
            rejected_edits.append(edit)
            continue
        line_text = lines[edit.line_id].text
        if edit.end_char > len(line_text):
            errors.append(
                f"Edit range exceeds line length for line {edit.line_id}. "
                f"line_length={len(line_text)} line={line_text!r} source={edit.source_text!r} "
                f"replacement={edit.replacement_text!r} range=[{edit.start_char}, {edit.end_char})."
            )
            rejected_edits.append(edit)
            continue
        if line_text[edit.start_char : edit.end_char] != edit.source_text:
            actual_slice = line_text[
                edit.start_char : min(edit.end_char, len(line_text))
            ]
            errors.append(
                f"Edit source text does not match line {edit.line_id} at range [{edit.start_char}, {edit.end_char}). "
                f"line={line_text!r} claimed_source={edit.source_text!r} actual_slice={actual_slice!r} "
                f"replacement={edit.replacement_text!r}."
            )
            rejected_edits.append(edit)
            continue
        if edit.source_text == edit.replacement_text:
            continue

        provisional.setdefault(edit.line_id, []).append(
            (
                edit,
                _ValidatedEdit(
                    source_text=edit.source_text,
                    replacement_text=edit.replacement_text,
                    start_char=edit.start_char,
                    end_char=edit.end_char,
                ),
            )
        )

    for line_id, line_edits in provisional.items():
        line_edits.sort(key=lambda item: (item[1].start_char, item[1].end_char))
        last_end = 0
        base_edits = list(existing_grouped_edits.get(line_id, []))
        non_overlapping_edits: list[tuple[NormalizationEdit, _ValidatedEdit]] = []
        for original_edit, item in line_edits:
            overlapping_base = next(
                (
                    base_item
                    for base_item in base_edits
                    if not (
                        item.end_char <= base_item.start_char
                        or item.start_char >= base_item.end_char
                    )
                ),
                None,
            )
            if overlapping_base is not None:
                errors.append(
                    f"Edit overlaps an already accepted edit on line {line_id}. "
                    f"line={lines[line_id].text!r} source={item.source_text!r} "
                    f"replacement={item.replacement_text!r} range=[{item.start_char}, {item.end_char}) "
                    f"accepted_range=[{overlapping_base.start_char}, {overlapping_base.end_char}) "
                    f"accepted_replacement={overlapping_base.replacement_text!r}."
                )
                rejected_edits.append(original_edit)
                continue
            if item.start_char < last_end:
                errors.append(
                    f"Overlapping edits returned for line {line_id}. "
                    f"line={lines[line_id].text!r} source={item.source_text!r} replacement={item.replacement_text!r} "
                    f"range=[{item.start_char}, {item.end_char})."
                )
                rejected_edits.append(original_edit)
                last_end = max(last_end, item.end_char)
                continue
            last_end = item.end_char
            non_overlapping_edits.append((original_edit, item))

        if not non_overlapping_edits:
            continue

        reverse_result = _validate_line_edit_contexts(
            line=lines[line_id],
            line_id=line_id,
            line_edits=non_overlapping_edits,
            allowed_terms=allowed_terms,
            base_edits=base_edits,
            reverse=True,
        )
        chosen_result = reverse_result
        if len(non_overlapping_edits) > 1:
            forward_result = _validate_line_edit_contexts(
                line=lines[line_id],
                line_id=line_id,
                line_edits=non_overlapping_edits,
                allowed_terms=allowed_terms,
                base_edits=base_edits,
                reverse=False,
            )
            if len(forward_result.errors) < len(reverse_result.errors):
                chosen_result = forward_result

        errors.extend(chosen_result.errors)
        rejected_edits.extend(chosen_result.rejected_edits)
        if chosen_result.accepted_edits:
            grouped.setdefault(line_id, []).extend(chosen_result.accepted_edits)
    return _ValidationResult(grouped, errors, rejected_edits)


def _validate_line_edit_contexts(
    *,
    line: SubtitleLine,
    line_id: int,
    line_edits: list[tuple[NormalizationEdit, _ValidatedEdit]],
    allowed_terms: set[str],
    base_edits: list[_ValidatedEdit],
    reverse: bool,
) -> _ContextValidationResult:
    ordered_line_edits = sorted(
        line_edits,
        key=lambda item: (item[1].start_char, item[1].end_char),
        reverse=reverse,
    )
    accepted_edits: list[_ValidatedEdit] = []
    errors: list[str] = []
    rejected_edits: list[NormalizationEdit] = []
    context_base_edits = list(base_edits)

    for original_edit, item in ordered_line_edits:
        context_before, context_start, context_end = _context_text_for_edit(
            line.text,
            item,
            base_edits=context_base_edits,
        )
        if not _replacement_matches_allowed_term_in_context(
            context_before,
            item,
            allowed_terms=allowed_terms,
            context_start=context_start,
            context_end=context_end,
        ):
            line_after_edit = (
                context_before[:context_start]
                + item.replacement_text
                + context_before[context_end:]
            )
            errors.append(
                f"Replacement {item.replacement_text!r} does not match any approved term in context "
                f"for line {line_id}. line_before={context_before!r} line_after={line_after_edit!r} "
                f"source={item.source_text!r} range=[{item.start_char}, {item.end_char}) "
                f"approved_terms={sorted(allowed_terms)!r}."
            )
            rejected_edits.append(original_edit)
            continue
        accepted_edits.append(item)
        context_base_edits.append(item)

    return _ContextValidationResult(
        accepted_edits=accepted_edits,
        errors=errors,
        rejected_edits=rejected_edits,
    )


def _replacement_matches_allowed_term_in_context(
    line_text: str,
    edit: _ValidatedEdit,
    *,
    allowed_terms: set[str],
    context_start: int | None = None,
    context_end: int | None = None,
) -> bool:
    if edit.replacement_text in allowed_terms:
        return True

    if context_start is None:
        context_start = edit.start_char
    if context_end is None:
        context_end = edit.end_char

    line_after_edit = (
        line_text[:context_start] + edit.replacement_text + line_text[context_end:]
    )
    replaced_start = context_start
    replaced_end = context_start + len(edit.replacement_text)

    for term in allowed_terms:
        for match_start, match_end, _ in _find_relaxed_term_matches(
            line_after_edit, term
        ):
            if max(match_start, replaced_start) < min(match_end, replaced_end):
                return True

    return False


def _find_source_occurrences(text: str, source_text: str) -> list[tuple[int, int]]:
    if not source_text:
        return []

    matches: list[tuple[int, int]] = []
    pos = 0
    while True:
        idx = text.find(source_text, pos)
        if idx == -1:
            break
        matches.append((idx, idx + len(source_text)))
        pos = idx + 1
    return matches


def _find_relaxed_term_matches(
    text: str,
    term: str,
) -> list[tuple[int, int, list[int]]]:
    if not text or not term:
        return []

    matches: list[tuple[int, int, list[int]]] = []
    for start in range(len(text)):
        if text[start] != term[0]:
            continue

        text_pos = start
        term_pos = 0
        skipped_positions: list[int] = []
        while text_pos < len(text) and term_pos < len(term):
            current = text[text_pos]
            if current == term[term_pos]:
                text_pos += 1
                term_pos += 1
                continue
            if current in IGNORABLE_CONTEXT_PUNCTUATION:
                skipped_positions.append(text_pos)
                text_pos += 1
                continue
            break

        if term_pos == len(term):
            matches.append((start, text_pos, skipped_positions))
    return matches


def _map_original_pos_to_replaced(
    original_pos: int,
    spans: list[ReplacementSpan],
) -> int:
    offset = 0
    for span in spans:
        if span.orig_start < original_pos < span.orig_end:
            raise ValueError(
                f"Original position {original_pos} falls inside accepted edit range "
                f"[{span.orig_start}, {span.orig_end})."
            )
        if span.orig_end <= original_pos:
            offset += (span.replaced_end - span.replaced_start) - (
                span.orig_end - span.orig_start
            )
    return original_pos + offset


def _context_text_for_edit(
    line_text: str,
    edit: _ValidatedEdit,
    *,
    base_edits: list[_ValidatedEdit],
) -> tuple[str, int, int]:
    if not base_edits:
        return line_text, edit.start_char, edit.end_char

    context_text, spans = _apply_line_edits_with_spans(line_text, base_edits)
    return (
        context_text,
        _map_original_pos_to_replaced(edit.start_char, spans),
        _map_original_pos_to_replaced(edit.end_char, spans),
    )


def _apply_line_edits_with_mapping(
    text: str,
    edits: list[_ValidatedEdit],
) -> tuple[str, list[ReplacementSpan], list[int | None]]:
    if not edits:
        return text, [], list(range(len(text)))

    edits = sorted(edits, key=lambda item: (item.start_char, item.end_char))
    result_parts: list[str] = []
    spans: list[ReplacementSpan] = []
    replaced_to_orig: list[int | None] = []
    orig_pos = 0
    replaced_pos = 0

    for edit in edits:
        if edit.start_char < orig_pos:
            raise ValueError("Overlapping edits are not allowed.")
        if text[edit.start_char : edit.end_char] != edit.source_text:
            raise ValueError("Edit source text does not match the provided span.")

        if edit.start_char > orig_pos:
            chunk = text[orig_pos : edit.start_char]
            result_parts.append(chunk)
            replaced_to_orig.extend(range(orig_pos, edit.start_char))
            replaced_pos += len(chunk)

        result_parts.append(edit.replacement_text)
        replaced_to_orig.extend([None] * len(edit.replacement_text))
        spans.append(
            ReplacementSpan(
                orig_start=edit.start_char,
                orig_end=edit.end_char,
                replaced_start=replaced_pos,
                replaced_end=replaced_pos + len(edit.replacement_text),
            )
        )
        replaced_pos += len(edit.replacement_text)
        orig_pos = edit.end_char

    if orig_pos < len(text):
        chunk = text[orig_pos:]
        result_parts.append(chunk)
        replaced_to_orig.extend(range(orig_pos, len(text)))

    return "".join(result_parts), spans, replaced_to_orig


def _override_edit_ranges_best_effort(
    lines: list[SubtitleLine],
    edits: list[NormalizationEdit],
) -> _RangeOverrideResult:
    resolved: list[NormalizationEdit] = []
    unresolved: list[NormalizationEdit] = []
    errors: list[str] = []

    for edit in edits:
        if not 0 <= edit.line_id < len(lines):
            errors.append(
                f"Edit has invalid line_id {edit.line_id} for source={edit.source_text!r} "
                f"replacement={edit.replacement_text!r} claimed_range=[{edit.start_char}, {edit.end_char})."
            )
            unresolved.append(edit)
            continue

        line_text = lines[edit.line_id].text
        matches = _find_source_occurrences(line_text, edit.source_text)
        if not matches:
            errors.append(
                f"Edit source text could not be found in line {edit.line_id}. "
                f"line={line_text!r} source={edit.source_text!r} replacement={edit.replacement_text!r} "
                f"claimed_range=[{edit.start_char}, {edit.end_char})."
            )
            unresolved.append(edit)
            continue

        start_char, end_char = min(
            matches,
            key=lambda match: (abs(match[0] - edit.start_char), match[0]),
        )
        if start_char != edit.start_char or end_char != edit.end_char:
            logger.warning(
                "Overriding edit range for line %s source=%r replacement=%r from [%s, %s) to [%s, %s).",
                edit.line_id,
                edit.source_text,
                edit.replacement_text,
                edit.start_char,
                edit.end_char,
                start_char,
                end_char,
            )
        resolved.append(
            NormalizationEdit(
                line_id=edit.line_id,
                source_text=edit.source_text,
                replacement_text=edit.replacement_text,
                start_char=start_char,
                end_char=end_char,
            )
        )

    return _RangeOverrideResult(
        resolved_edits=resolved,
        unresolved_edits=unresolved,
        errors=errors,
    )


def _override_retry_edit_ranges(
    lines: list[SubtitleLine],
    edits: list[NormalizationEdit],
) -> list[NormalizationEdit]:
    result = _override_edit_ranges_best_effort(lines, edits)
    if result.errors:
        retry_errors = [
            error.replace(
                "Edit has invalid line_id", "Retry edit has invalid line_id"
            ).replace(
                "Edit source text could not be found",
                "Retry edit source text could not be found",
            )
            for error in result.errors
        ]
        raise NormalizerValidationError(retry_errors)
    return result.resolved_edits


def _validated_to_normalization_edits(
    grouped_edits: dict[int, list[_ValidatedEdit]],
) -> list[NormalizationEdit]:
    result: list[NormalizationEdit] = []
    for line_id, line_edits in grouped_edits.items():
        for edit in line_edits:
            result.append(
                NormalizationEdit(
                    line_id=line_id,
                    source_text=edit.source_text,
                    replacement_text=edit.replacement_text,
                    start_char=edit.start_char,
                    end_char=edit.end_char,
                )
            )
    result.sort(key=lambda item: (item.line_id, item.start_char, item.end_char))
    return result


def _merge_grouped_validated_edits(
    first: dict[int, list[_ValidatedEdit]],
    second: dict[int, list[_ValidatedEdit]],
) -> dict[int, list[_ValidatedEdit]]:
    merged: dict[int, list[_ValidatedEdit]] = {
        line_id: list(line_edits) for line_id, line_edits in first.items()
    }
    for line_id, line_edits in second.items():
        merged.setdefault(line_id, []).extend(line_edits)

    for line_id, line_edits in merged.items():
        line_edits.sort(key=lambda item: (item.start_char, item.end_char))
        last_end = 0
        for item in line_edits:
            if item.start_char < last_end:
                raise NormalizerValidationError(
                    [
                        f"Correction edits overlap existing valid edits on line {line_id}."
                    ]
                )
            last_end = item.end_char
    return merged


def _build_cleanup_deletion_edits(
    line_text: str,
    edits: list[_ValidatedEdit],
    *,
    allowed_terms: set[str],
) -> list[_ValidatedEdit]:
    if not edits:
        return []

    replaced_text, _, replaced_to_orig = _apply_line_edits_with_mapping(
        line_text, edits
    )
    occupied_orig_ranges = [(edit.start_char, edit.end_char) for edit in edits]
    orig_indices_to_delete: set[int] = set()

    for term in allowed_terms:
        for _, _, skipped_positions in _find_relaxed_term_matches(replaced_text, term):
            for position in skipped_positions:
                if not (0 <= position < len(replaced_to_orig)):
                    continue
                orig_index = replaced_to_orig[position]
                if orig_index is None:
                    continue
                char = line_text[orig_index]
                if char not in IGNORABLE_CONTEXT_PUNCTUATION:
                    continue
                if any(
                    start <= orig_index < end for start, end in occupied_orig_ranges
                ):
                    continue
                orig_indices_to_delete.add(orig_index)

    if not orig_indices_to_delete:
        return []

    sorted_indices = sorted(orig_indices_to_delete)
    cleanup_edits: list[_ValidatedEdit] = []
    run_start = sorted_indices[0]
    run_end = run_start + 1
    for index in sorted_indices[1:]:
        if index == run_end:
            run_end += 1
            continue
        cleanup_edits.append(
            _ValidatedEdit(
                source_text=line_text[run_start:run_end],
                replacement_text="",
                start_char=run_start,
                end_char=run_end,
            )
        )
        run_start = index
        run_end = index + 1
    cleanup_edits.append(
        _ValidatedEdit(
            source_text=line_text[run_start:run_end],
            replacement_text="",
            start_char=run_start,
            end_char=run_end,
        )
    )
    return cleanup_edits


def apply_llm_normalization(
    lines: list[SubtitleLine],
    config: dict,
) -> list[SubtitleLine]:
    raw_terms = config.get("terms", [])
    terms = [NormalizerTerm.model_validate(item) for item in raw_terms]
    if not terms:
        raise ValueError("LLM normalizer requires at least one configured term.")

    normalizer = LLMKeywordNormalizer(
        project_id=config.get("project_id"),
        model=config.get("model"),
        location=config.get("location", "global"),
        provider=config.get("provider", "google-vertex"),
        reasoning_effort=config.get("reasoning_effort", ReasoningEffort.MINIMAL),
        reasoning_budget_tokens=config.get("reasoning_budget_tokens"),
        reasoning_dynamic=config.get("reasoning_dynamic"),
        provider_options=config.get("provider_options"),
        trace_path=config.get("llm_trace_path"),
    )
    edits = normalizer.propose_edits(lines, terms)
    allowed_terms = {term.value for term in terms}
    validation = _collect_llm_edit_validation(
        lines,
        edits,
        allowed_terms=allowed_terms,
    )
    grouped_edits = validation.grouped_edits
    if validation.errors:
        if not config.get("allow_llm_correction"):
            _log_validation_errors(
                logging.ERROR,
                "LLM normalizer first attempt",
                validation.errors,
            )
            raise NormalizerValidationError(validation.errors)
        locally_repaired = _override_edit_ranges_best_effort(
            lines, validation.rejected_edits
        )
        remaining_errors = list(locally_repaired.errors)
        remaining_rejected_edits = list(locally_repaired.unresolved_edits)

        if locally_repaired.resolved_edits:
            repaired_validation = _collect_llm_edit_validation(
                lines,
                locally_repaired.resolved_edits,
                allowed_terms=allowed_terms,
                existing_grouped_edits=grouped_edits,
            )
            if repaired_validation.grouped_edits:
                grouped_edits = _merge_grouped_validated_edits(
                    grouped_edits,
                    repaired_validation.grouped_edits,
                )
                logger.info(
                    "Locally repaired %s rejected first-pass edits before retrying the normalizer.",
                    sum(
                        len(line_edits)
                        for line_edits in repaired_validation.grouped_edits.values()
                    ),
                )
            remaining_errors.extend(repaired_validation.errors)
            remaining_rejected_edits.extend(repaired_validation.rejected_edits)

        if not remaining_errors:
            validation = _ValidationResult(grouped_edits, [], [])
        else:
            _log_validation_errors(
                logging.WARNING,
                "LLM normalizer first attempt",
                remaining_errors,
            )
            logger.warning(
                "LLM normalizer produced invalid edits after local range repair; requesting one correction pass."
            )
        corrected_edits = (
            normalizer.propose_corrected_edits(
                lines,
                terms,
                accepted_edits=_validated_to_normalization_edits(grouped_edits),
                previous_edits=remaining_rejected_edits,
                validation_errors=remaining_errors,
            )
            if remaining_errors
            else []
        )
        if corrected_edits:
            try:
                corrected_edits = _override_retry_edit_ranges(lines, corrected_edits)
            except NormalizerValidationError as exc:
                _log_validation_errors(
                    logging.ERROR,
                    "LLM normalizer correction attempt",
                    exc.errors,
                )
                raise
            corrected_validation = _collect_llm_edit_validation(
                lines,
                corrected_edits,
                allowed_terms=allowed_terms,
                existing_grouped_edits=grouped_edits,
            )
            if corrected_validation.errors:
                _log_validation_errors(
                    logging.ERROR,
                    "LLM normalizer correction attempt",
                    corrected_validation.errors,
                )
                raise NormalizerValidationError(corrected_validation.errors)
            grouped_edits = _merge_grouped_validated_edits(
                grouped_edits,
                corrected_validation.grouped_edits,
            )

    if not grouped_edits:
        logger.info("LLM normalizer proposed no edits.")
        return lines

    for line_id, line_edits in list(grouped_edits.items()):
        cleanup_edits = _build_cleanup_deletion_edits(
            lines[line_id].text,
            line_edits,
            allowed_terms=allowed_terms,
        )
        if cleanup_edits:
            grouped_edits = _merge_grouped_validated_edits(
                grouped_edits,
                {line_id: cleanup_edits},
            )

    logger.info(
        "Applying %s LLM-proposed normalization edits across %s lines...",
        sum(len(line_edits) for line_edits in grouped_edits.values()),
        len(grouped_edits),
    )
    for line_id, line_edits in grouped_edits.items():
        new_text, spans = _apply_line_edits_with_spans(lines[line_id].text, line_edits)
        lines[line_id].text = new_text
        lines[line_id].replacement_spans = spans
    return lines


def apply_normalization(
    lines: list[SubtitleLine],
    config: dict | None,
) -> list[SubtitleLine]:
    if not config:
        return lines

    engine = str(config.get("engine", "exact")).lower()
    if engine == "exact":
        replacements = config.get("replacements", {})
        if not isinstance(replacements, dict):
            raise ValueError("Exact normalizer requires a replacements dictionary.")
        return apply_exact_normalization(lines, replacements)
    if engine == "llm":
        return apply_llm_normalization(lines, config)
    raise ValueError(f"Unknown format normalizer engine: {engine}")
