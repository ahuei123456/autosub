"""Tests for the combined radio_discourse + corners classifier path."""

from unittest.mock import patch

import pytest

from autosub.core.schemas import SubtitleLine
from autosub.pipeline.format.main import _apply_combined_extensions


def _line(text, start=0.0, end=1.0, speaker=None):
    return SubtitleLine(
        text=text, start_time=start, end_time=end, speaker=speaker
    )


@pytest.fixture
def sample_lines():
    return [
        _line("皆さんこんにちは！リエラジへようこそ！", 0, 3),
        _line("今日もお便りをいただいています。", 3, 6),
        _line("ラジオネームさくらさんから。", 6, 9),
        _line("先日のライブ最高でした！", 9, 12),
        _line("ありがとうね。嬉しい。", 12, 15),
        _line("続いてのコーナー、投書箱に参りましょう。", 15, 18),
        _line("最初のお題はこちらです。", 18, 21),
    ]


@pytest.fixture
def radio_config():
    return {
        "enabled": True,
        "engine": "hybrid",
        "split_framing_phrases": True,
        "label_roles": True,
    }


@pytest.fixture
def corners_config():
    return {
        "enabled": True,
        "engine": "hybrid",
        "segments": [
            {"name": "Opening", "cues": ["ようこそ"]},
            {"name": "Suggestion Box", "cues": ["投書箱"]},
        ],
    }


class TestApplyCombinedExtensions:
    def test_falls_back_to_rules_and_cues_on_vertex_error(
        self, sample_lines, radio_config, corners_config, tmp_path
    ):
        """When LLM fails with hybrid engines, fallback to rules + cues."""
        from autosub.core.errors import VertexError

        output_path = tmp_path / "test.ass"

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=VertexError("test error"),
        ):
            result = _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Should not raise — hybrid engines allow fallback
        assert len(result) >= len(sample_lines)  # may have more due to framing splits
        # Cue-based corners should be detected
        corners = [line.corner for line in result if line.corner]
        assert any("Opening" in c for c in corners)

    def test_raises_on_vertex_error_with_strict_engine(
        self, sample_lines, corners_config, tmp_path
    ):
        """When radio engine is 'llm' (strict), VertexError should propagate."""
        from autosub.core.errors import VertexError

        radio_config = {
            "enabled": True,
            "engine": "llm",
            "split_framing_phrases": True,
            "label_roles": True,
        }
        output_path = tmp_path / "test.ass"

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=VertexError("test error"),
        ):
            with pytest.raises(VertexError):
                _apply_combined_extensions(
                    sample_lines, radio_config, corners_config, output_path
                )

    def test_successful_combined_classification(
        self, sample_lines, radio_config, corners_config, tmp_path
    ):
        """When LLM succeeds, uses its roles and corners."""
        output_path = tmp_path / "test.ass"

        def mock_classify(lines, fallback_roles, segments, config):
            roles = ["host"] * len(lines)
            corners = [None] * len(lines)
            corners[0] = "Opening"
            return roles, corners

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            result = _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Should have roles assigned
        roles = [line.role for line in result]
        assert all(r == "host" for r in roles)

        # Should have the LLM corner + cue corners merged
        corners = [line.corner for line in result]
        assert "Opening" in corners

    def test_merges_corners_config_into_combined(
        self, sample_lines, tmp_path
    ):
        """Corners config settings fill gaps not specified by radio config."""
        radio_config = {
            "enabled": True,
            "engine": "hybrid",
            "split_framing_phrases": True,
            "label_roles": True,
        }
        corners_config = {
            "enabled": True,
            "engine": "hybrid",
            "model": "corners-specific-model",
            "location": "us-central1",
            "segments": [
                {"name": "Opening", "cues": ["ようこそ"]},
            ],
        }
        output_path = tmp_path / "test.ass"

        captured_config = {}

        def mock_classify(lines, fallback_roles, segments, config):
            captured_config.update(config)
            return ["host"] * len(lines), [None] * len(lines)

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Corners-specific settings should be present since radio doesn't set them
        assert captured_config["model"] == "corners-specific-model"
        assert captured_config["location"] == "us-central1"

    def test_radio_config_takes_precedence_over_corners(
        self, sample_lines, tmp_path
    ):
        """When both configs specify the same key, radio config wins."""
        radio_config = {
            "enabled": True,
            "engine": "hybrid",
            "model": "radio-model",
            "split_framing_phrases": True,
            "label_roles": True,
        }
        corners_config = {
            "enabled": True,
            "engine": "hybrid",
            "model": "corners-model",
            "segments": [
                {"name": "Opening", "cues": ["ようこそ"]},
            ],
        }
        output_path = tmp_path / "test.ass"

        captured_config = {}

        def mock_classify(lines, fallback_roles, segments, config):
            captured_config.update(config)
            return ["host"] * len(lines), [None] * len(lines)

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        assert captured_config["model"] == "radio-model"

    def test_cue_corners_fill_llm_gaps(
        self, sample_lines, radio_config, corners_config, tmp_path
    ):
        """Cue-detected corners fill in where LLM returns None."""
        output_path = tmp_path / "test.ass"

        def mock_classify(lines, fallback_roles, segments, config):
            roles = ["host"] * len(lines)
            corners = [None] * len(lines)
            # LLM doesn't detect "Suggestion Box" but cues will
            return roles, corners

        with patch(
            "autosub.extensions.combined_classifier.classify_combined",
            side_effect=mock_classify,
        ):
            result = _apply_combined_extensions(
                sample_lines, radio_config, corners_config, output_path
            )

        # Cue-based detection should find both corners
        corners = [line.corner for line in result if line.corner]
        assert "Opening" in corners
        assert "Suggestion Box" in corners
