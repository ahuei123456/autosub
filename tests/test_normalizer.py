from autosub.core.schemas import ReplacementSpan, SubtitleLine
from autosub.pipeline.format.normalizer import (
    LLMKeywordNormalizer,
    NormalizerValidationError,
    NormalizationEdit,
    apply_normalization,
)


def test_apply_normalization_exact_preserves_replacement_spans():
    lines = [SubtitleLine(text="の番は", start_time=0.0, end_time=1.0)]

    result = apply_normalization(
        lines,
        {
            "engine": "exact",
            "replacements": {"の番は": "のんばんは"},
        },
    )

    assert result[0].text == "のんばんは"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=3, replaced_start=0, replaced_end=5)
    ]


def test_apply_normalization_llm_applies_validated_edits_and_spans(monkeypatch):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        assert terms[0].value == "鈴原希実"
        assert terms[0].explanation == "Host name."
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=5,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
        },
    )

    assert result[0].text == "鈴原希実です"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=5, replaced_start=0, replaced_end=4)
    ]


def test_apply_normalization_llm_rejects_noncanonical_replacement(monkeypatch):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原のぞみ",
                start_char=0,
                end_char=5,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    try:
        apply_normalization(
            lines,
            {
                "engine": "llm",
                "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
            },
        )
    except ValueError as exc:
        assert "does not match any approved term in context" in str(exc)
        assert "line_before=" in str(exc)
        assert "approved_terms=" in str(exc)
    else:
        raise AssertionError("Expected ValueError for noncanonical replacement.")


def test_apply_normalization_llm_can_request_one_correction_pass(monkeypatch):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原のぞみ",
                start_char=0,
                end_char=5,
            )
        ]

    def fake_correct(
        self,
        lines,
        terms,
        *,
        accepted_edits,
        previous_edits,
        validation_errors,
    ):
        assert accepted_edits == []
        assert previous_edits[0].replacement_text == "鈴原のぞみ"
        assert any(
            "does not match any approved term in context" in item
            for item in validation_errors
        )
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原希実",
                start_char=3,
                end_char=8,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)
    monkeypatch.setattr(LLMKeywordNormalizer, "propose_corrected_edits", fake_correct)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "allow_llm_correction": True,
            "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
        },
    )

    assert result[0].text == "鈴原希実です"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=5, replaced_start=0, replaced_end=4)
    ]


def test_apply_normalization_llm_still_fails_when_correction_is_disabled(monkeypatch):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原のぞみ",
                start_char=0,
                end_char=5,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    try:
        apply_normalization(
            lines,
            {
                "engine": "llm",
                "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
            },
        )
    except NormalizerValidationError as exc:
        assert "does not match any approved term in context" in str(exc)
        assert "line_before=" in str(exc)
    else:
        raise AssertionError(
            "Expected NormalizerValidationError when correction is disabled."
        )


def test_apply_normalization_logs_first_attempt_errors_with_line_context(
    monkeypatch, caplog
):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原希実",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=4,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    with caplog.at_level("ERROR"):
        try:
            apply_normalization(
                lines,
                {
                    "engine": "llm",
                    "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
                },
            )
        except NormalizerValidationError:
            pass
        else:
            raise AssertionError("Expected NormalizerValidationError.")

    assert "LLM normalizer first attempt validation errors" in caplog.text
    assert "line='鈴原のソミです'" in caplog.text
    assert "claimed_source='鈴原希実'" in caplog.text
    assert "actual_slice='鈴原のソ'" in caplog.text


def test_apply_normalization_logs_second_attempt_errors_with_line_context(
    monkeypatch, caplog
):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原のぞみ",
                start_char=0,
                end_char=5,
            )
        ]

    def fake_correct(
        self,
        lines,
        terms,
        *,
        accepted_edits,
        previous_edits,
        validation_errors,
    ):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原希実",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=4,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)
    monkeypatch.setattr(LLMKeywordNormalizer, "propose_corrected_edits", fake_correct)

    with caplog.at_level("WARNING"):
        try:
            apply_normalization(
                lines,
                {
                    "engine": "llm",
                    "allow_llm_correction": True,
                    "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
                },
            )
        except NormalizerValidationError:
            pass
        else:
            raise AssertionError("Expected NormalizerValidationError.")

    assert "LLM normalizer first attempt validation errors" in caplog.text
    assert "LLM normalizer correction attempt validation errors" in caplog.text
    assert "line='鈴原のソミです'" in caplog.text
    assert "source='鈴原希実'" in caplog.text
    assert "claimed_range=[0, 4)" in caplog.text


def test_apply_normalization_llm_correction_only_retries_rejected_edits(monkeypatch):
    lines = [
        SubtitleLine(text="鈴原のソミとの番は", start_time=0.0, end_time=1.0),
    ]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=5,
            ),
            NormalizationEdit(
                line_id=0,
                source_text="の番は",
                replacement_text="こんばんは",
                start_char=6,
                end_char=9,
            ),
        ]

    def fake_correct(
        self,
        lines,
        terms,
        *,
        accepted_edits,
        previous_edits,
        validation_errors,
    ):
        assert len(accepted_edits) == 1
        assert accepted_edits[0].replacement_text == "鈴原希実"
        assert len(previous_edits) == 1
        assert previous_edits[0].replacement_text == "こんばんは"
        assert any(
            "does not match any approved term in context" in item
            for item in validation_errors
        )
        return [
            NormalizationEdit(
                line_id=0,
                source_text="の番は",
                replacement_text="のんばんは",
                start_char=2,
                end_char=5,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)
    monkeypatch.setattr(LLMKeywordNormalizer, "propose_corrected_edits", fake_correct)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "allow_llm_correction": True,
            "terms": [
                {"value": "鈴原希実", "explanation": "Host name."},
                {"value": "のんばんは", "explanation": "Show greeting."},
            ],
        },
    )

    assert result[0].text == "鈴原希実とのんばんは"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=5, replaced_start=0, replaced_end=4),
        ReplacementSpan(orig_start=6, orig_end=9, replaced_start=5, replaced_end=10),
    ]


def test_apply_normalization_llm_retry_still_fails_when_source_text_cannot_be_found(
    monkeypatch,
):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原のぞみ",
                start_char=0,
                end_char=5,
            )
        ]

    def fake_correct(
        self,
        lines,
        terms,
        *,
        accepted_edits,
        previous_edits,
        validation_errors,
    ):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原希実",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=4,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)
    monkeypatch.setattr(LLMKeywordNormalizer, "propose_corrected_edits", fake_correct)

    try:
        apply_normalization(
            lines,
            {
                "engine": "llm",
                "allow_llm_correction": True,
                "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
            },
        )
    except NormalizerValidationError as exc:
        assert "Retry edit source text could not be found" in str(exc)
        assert "line='鈴原のソミです'" in str(exc)
        assert "source='鈴原希実'" in str(exc)
    else:
        raise AssertionError(
            "Expected NormalizerValidationError when retry source text is absent."
        )


def test_apply_normalization_llm_repairs_first_pass_ranges_before_retry(monkeypatch):
    lines = [SubtitleLine(text="鈴原のソミです", start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原希実",
                start_char=3,
                end_char=8,
            )
        ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Correction pass should not run when local range repair succeeds."
        )

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)
    monkeypatch.setattr(LLMKeywordNormalizer, "propose_corrected_edits", fail_if_called)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "allow_llm_correction": True,
            "terms": [{"value": "鈴原希実", "explanation": "Host name."}],
        },
    )

    assert result[0].text == "鈴原希実です"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=5, replaced_start=0, replaced_end=4)
    ]


def test_apply_normalization_llm_retry_uses_accepted_edits_as_context(monkeypatch):
    line_text = "鈴原のソミの明日は何視聴と?"
    lines = [SubtitleLine(text=line_text, start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=5,
            ),
            NormalizationEdit(
                line_id=0,
                source_text="何視聴と",
                replacement_text="こんばんは",
                start_char=9,
                end_char=13,
            ),
        ]

    def fake_correct(
        self,
        lines,
        terms,
        *,
        accepted_edits,
        previous_edits,
        validation_errors,
    ):
        assert len(accepted_edits) == 1
        assert accepted_edits[0].replacement_text == "鈴原希実"
        assert len(previous_edits) == 1
        assert previous_edits[0].source_text == "何視聴と"
        return [
            NormalizationEdit(
                line_id=0,
                source_text="何視聴と",
                replacement_text="なんしちょっと",
                start_char=9,
                end_char=13,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)
    monkeypatch.setattr(LLMKeywordNormalizer, "propose_corrected_edits", fake_correct)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "allow_llm_correction": True,
            "terms": [
                {"value": "鈴原希実", "explanation": "Host name."},
                {
                    "value": "鈴原希実の明日はなんしちょっと",
                    "explanation": "Program title phrase.",
                },
            ],
        },
    )

    assert result[0].text == "鈴原希実の明日はなんしちょっと?"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=5, replaced_start=0, replaced_end=4),
        ReplacementSpan(orig_start=9, orig_end=13, replaced_start=8, replaced_end=15),
    ]


def test_apply_normalization_llm_relaxes_context_match_and_removes_punctuation(
    monkeypatch,
):
    line_text = "鈴原のソミの明日は、何しちょっと?"
    source_text = "何しちょっと"
    start_char = line_text.index(source_text)
    end_char = start_char + len(source_text)
    comma_index = line_text.index("、")
    lines = [SubtitleLine(text=line_text, start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="鈴原のソミ",
                replacement_text="鈴原希実",
                start_char=0,
                end_char=5,
            ),
            NormalizationEdit(
                line_id=0,
                source_text=source_text,
                replacement_text="なんしちょっと",
                start_char=start_char,
                end_char=end_char,
            ),
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "terms": [
                {"value": "鈴原希実", "explanation": "Host name."},
                {
                    "value": "鈴原希実の明日はなんしちょっと",
                    "explanation": "Program title phrase.",
                },
            ],
        },
    )

    assert result[0].text == "鈴原希実の明日はなんしちょっと?"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=5, replaced_start=0, replaced_end=4),
        ReplacementSpan(
            orig_start=comma_index,
            orig_end=comma_index + 1,
            replaced_start=8,
            replaced_end=8,
        ),
        ReplacementSpan(
            orig_start=start_char, orig_end=end_char, replaced_start=8, replaced_end=15
        ),
    ]


def test_apply_normalization_llm_allows_partial_replacement_when_it_forms_approved_term(
    monkeypatch,
):
    line_text = "鈴原希実の明日はなんしちゃっとです"
    source_text = "なんしちゃっと"
    start_char = line_text.index(source_text)
    end_char = start_char + len(source_text)
    lines = [
        SubtitleLine(
            text=line_text,
            start_time=0.0,
            end_time=1.0,
        )
    ]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text=source_text,
                replacement_text="なんしちょっと",
                start_char=start_char,
                end_char=end_char,
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "terms": [
                {
                    "value": "鈴原希実の明日はなんしちょっと",
                    "explanation": "Program title phrase.",
                }
            ],
        },
    )

    assert result[0].text == "鈴原希実の明日はなんしちょっとです"
    assert result[0].replacement_spans == [
        ReplacementSpan(
            orig_start=start_char,
            orig_end=end_char,
            replaced_start=start_char,
            replaced_end=start_char + len("なんしちょっと"),
        )
    ]


def test_apply_normalization_llm_allows_replacement_text_with_trailing_punctuation(
    monkeypatch,
):
    line_text = "鈴原望みの明日は何しちゃった?"
    lines = [SubtitleLine(text=line_text, start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text=line_text,
                replacement_text="鈴原希実の明日はなんしちょっと?",
                start_char=0,
                end_char=len(line_text),
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "terms": [
                {
                    "value": "鈴原希実の明日はなんしちょっと",
                    "explanation": "Program title phrase.",
                }
            ],
        },
    )

    assert result[0].text == "鈴原希実の明日はなんしちょっと?"
    assert result[0].replacement_spans == [
        ReplacementSpan(
            orig_start=0,
            orig_end=len(line_text),
            replaced_start=0,
            replaced_end=len("鈴原希実の明日はなんしちょっと?"),
        )
    ]


def test_apply_normalization_llm_allows_replacement_text_with_context_after_term(
    monkeypatch,
):
    source_text = "おののね相沢さん"
    line_text = f"{source_text}からいただきました。"
    lines = [SubtitleLine(text=line_text, start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text=source_text,
                replacement_text="のんのんネーム相沢さん",
                start_char=0,
                end_char=len(source_text),
            )
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "terms": [
                {
                    "value": "のんのんネーム",
                    "explanation": "Listener name callout.",
                }
            ],
        },
    )

    assert result[0].text == "のんのんネーム相沢さんからいただきました。"
    assert result[0].replacement_spans == [
        ReplacementSpan(
            orig_start=0,
            orig_end=len(source_text),
            replaced_start=0,
            replaced_end=len("のんのんネーム相沢さん"),
        )
    ]


def test_apply_normalization_llm_can_validate_same_line_edits_back_to_front(
    monkeypatch,
):
    line_text = "fooXbar"
    lines = [SubtitleLine(text=line_text, start_time=0.0, end_time=1.0)]

    def fake_propose(self, lines, terms):
        return [
            NormalizationEdit(
                line_id=0,
                source_text="fooX",
                replacement_text="foo",
                start_char=0,
                end_char=4,
            ),
            NormalizationEdit(
                line_id=0,
                source_text="bar",
                replacement_text="BAR",
                start_char=4,
                end_char=7,
            ),
        ]

    monkeypatch.setattr(LLMKeywordNormalizer, "propose_edits", fake_propose)

    result = apply_normalization(
        lines,
        {
            "engine": "llm",
            "terms": [
                {"value": "BAR", "explanation": "Approved suffix."},
                {"value": "fooBAR", "explanation": "Approved combined phrase."},
            ],
        },
    )

    assert result[0].text == "fooBAR"
    assert result[0].replacement_spans == [
        ReplacementSpan(orig_start=0, orig_end=4, replaced_start=0, replaced_end=3),
        ReplacementSpan(orig_start=4, orig_end=7, replaced_start=3, replaced_end=6),
    ]
