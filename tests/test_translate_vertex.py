from autosub.pipeline.translate.vertex import VertexTranslator


def test_vertex_prompt_includes_line_ending_style_guidance():
    translator = VertexTranslator(
        project_id="test-project",
        source_lang="ja",
        target_lang="en",
        system_prompt="Keep the host warm and conversational.",
    )

    instruction = translator._get_system_instruction()

    assert (
        "Prefer ending subtitle lines on natural punctuation whenever possible"
        in instruction
    )
    assert "Move trailing connectives such as 'but', 'and', 'so'" in instruction
    assert "Speaker and style context:" in instruction
    assert "Keep the host warm and conversational." in instruction
