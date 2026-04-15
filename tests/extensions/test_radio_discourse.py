import pytest

from autosub.core.schemas import SubtitleLine, TranscribedWord
from autosub.extensions.radio_discourse.main import (
    apply_radio_discourse,
    split_host_meta_suffix,
)


def test_radio_discourse_splits_framing_phrase_and_labels_roles():
    lines = [
        SubtitleLine(
            text="メールを送るのは初めてです。",
            start_time=0.0,
            end_time=2.0,
        ),
        SubtitleLine(
            text="おお、嬉しい。",
            start_time=2.0,
            end_time=3.0,
        ),
        SubtitleLine(
            text="いろんなコーデに合わせられるのでおすすめですといただきました。",
            start_time=3.0,
            end_time=6.0,
        ),
    ]

    result = apply_radio_discourse(lines, {"enabled": True})

    assert [line.role for line in result] == [
        "listener_mail",
        "host",
        "listener_mail",
        "host_meta",
    ]
    assert result[2].text == "いろんなコーデに合わせられるのでおすすめです。"
    assert result[3].text == "といただきました。"
    assert result[2].end_time <= result[3].start_time


def test_split_host_meta_suffix_uses_word_timestamp():
    """
    Split time should snap to ね.end_time (589.86) rather than the old
    character-proportion estimate (~589.56 for this line).
    Real words extracted from nonshichotto/144/output.json.
    """
    words = [
        TranscribedWord(word="夢", start_time=587.82, end_time=588.1),
        TranscribedWord(word="の", start_time=588.1, end_time=588.22),
        TranscribedWord(word="国", start_time=588.22, end_time=588.5),
        TranscribedWord(word="行ける", start_time=588.5, end_time=588.86),
        TranscribedWord(word="と", start_time=588.86, end_time=589.02),
        TranscribedWord(word="いい", start_time=589.02, end_time=589.14),
        TranscribedWord(word="です", start_time=589.14, end_time=589.46),
        TranscribedWord(word="ね", start_time=589.46, end_time=589.86),
        TranscribedWord(word="と", start_time=589.86, end_time=589.94),
        TranscribedWord(word="いただき", start_time=589.94, end_time=590.42),
        TranscribedWord(word="まし", start_time=590.42, end_time=590.7),
        TranscribedWord(word="た。", start_time=590.7, end_time=590.86),
    ]
    line = SubtitleLine(
        text="夢の国行けるといいですねといただきました。",
        start_time=587.82,
        end_time=590.86,
        words=words,
    )

    result = split_host_meta_suffix(line)

    assert len(result) == 2
    assert result[0].text == "夢の国行けるといいですね。"
    assert result[1].text == "といただきました。"
    assert result[0].end_time == pytest.approx(589.86)
    assert result[1].start_time == pytest.approx(589.86)


def test_split_host_meta_suffix_words_partitioned():
    """Words before the split go to the main body; suffix words go to the suffix line."""
    words = [
        TranscribedWord(word="夢", start_time=587.82, end_time=588.1),
        TranscribedWord(word="ね", start_time=589.46, end_time=589.86),
        TranscribedWord(word="と", start_time=589.86, end_time=589.94),
        TranscribedWord(word="いただき", start_time=589.94, end_time=590.42),
        TranscribedWord(word="まし", start_time=590.42, end_time=590.7),
        TranscribedWord(word="た。", start_time=590.7, end_time=590.86),
    ]
    line = SubtitleLine(
        text="夢ねといただきました。",
        start_time=587.82,
        end_time=590.86,
        words=words,
    )

    result = split_host_meta_suffix(line)

    assert len(result) == 2
    # split_char_pos = len("夢ねといただきました。") - len("といただきました。") = 11 - 9 = 2
    # orig_pos = 2; 夢(1), ね(2) >= 2 → split_time = ね.end_time = 589.86
    assert result[0].end_time == pytest.approx(589.86)
    main_body_word_texts = [w.word for w in result[0].words]
    suffix_word_texts = [w.word for w in result[1].words]
    assert "夢" in main_body_word_texts
    assert "ね" in main_body_word_texts
    assert "と" in suffix_word_texts
    assert "た。" in suffix_word_texts


def test_split_host_meta_suffix_no_words_falls_back_gracefully():
    """With no words, split_host_meta_suffix should still produce two lines."""
    line = SubtitleLine(
        text="おすすめですといただきました。",
        start_time=0.0,
        end_time=3.0,
    )
    result = split_host_meta_suffix(line)
    assert len(result) == 2
    assert result[0].end_time <= result[1].start_time


def test_apply_radio_discourse_greetings_splits_before_extension():
    """greetings config splits lines after the greeting phrase, before role labeling."""
    lines = [
        SubtitleLine(
            text="のんばんは今日もよろしく。",
            start_time=0.0,
            end_time=3.0,
            words=[
                TranscribedWord(word="のん", start_time=0.0, end_time=0.5),
                TranscribedWord(word="ばん", start_time=0.5, end_time=0.8),
                TranscribedWord(word="は", start_time=0.8, end_time=1.0),
                TranscribedWord(word="今日も", start_time=1.1, end_time=2.0),
                TranscribedWord(word="よろしく。", start_time=2.0, end_time=3.0),
            ],
        ),
    ]

    result = apply_radio_discourse(
        lines,
        {"enabled": True, "greetings": ["のんばんは"], "split_framing_phrases": False},
    )

    assert len(result) == 2
    assert result[0].text == "のんばんは。"
    assert result[1].text == "今日もよろしく。"
    assert result[0].end_time == pytest.approx(1.0)  # は.end_time
    assert result[1].start_time == pytest.approx(1.0)


def test_apply_radio_discourse_greetings_multiple_phrases():
    """Multiple greetings are each split after."""
    lines = [
        SubtitleLine(text="のんばんは今日も。", start_time=0.0, end_time=4.0),
        SubtitleLine(text="おはようございます明日も。", start_time=4.0, end_time=8.0),
    ]

    result = apply_radio_discourse(
        lines,
        {
            "enabled": True,
            "greetings": ["のんばんは", "おはようございます"],
            "split_framing_phrases": False,
        },
    )

    assert len(result) == 4
    assert result[0].text == "のんばんは。"
    assert result[1].text == "今日も。"
    assert result[2].text == "おはようございます。"
    assert result[3].text == "明日も。"


def test_apply_radio_discourse_greetings_coerces_string_to_one_item_list():
    lines = [
        SubtitleLine(
            text="のんばんは今日もよろしく。",
            start_time=0.0,
            end_time=3.0,
            words=[
                TranscribedWord(word="のん", start_time=0.0, end_time=0.5),
                TranscribedWord(word="ばん", start_time=0.5, end_time=0.8),
                TranscribedWord(word="は", start_time=0.8, end_time=1.0),
                TranscribedWord(word="今日も", start_time=1.1, end_time=2.0),
                TranscribedWord(word="よろしく。", start_time=2.0, end_time=3.0),
            ],
        ),
    ]

    result = apply_radio_discourse(
        lines,
        {"enabled": True, "greetings": "のんばんは", "split_framing_phrases": False},
    )

    assert len(result) == 2
    assert result[0].text == "のんばんは。"
    assert result[1].text == "今日もよろしく。"
    assert result[0].end_time == pytest.approx(1.0)
    assert result[1].start_time == pytest.approx(1.0)


def test_apply_radio_discourse_greetings_keeps_trailing_period_with_greeting():
    lines = [
        SubtitleLine(
            text="のんばんは。のんばんは",
            start_time=0.0,
            end_time=2.0,
            words=[
                TranscribedWord(word="のん", start_time=0.0, end_time=0.3),
                TranscribedWord(word="ばん", start_time=0.3, end_time=0.6),
                TranscribedWord(word="は。", start_time=0.6, end_time=1.0),
                TranscribedWord(word="のん", start_time=1.0, end_time=1.3),
                TranscribedWord(word="ばん", start_time=1.3, end_time=1.6),
                TranscribedWord(word="は", start_time=1.6, end_time=2.0),
            ],
        ),
    ]

    result = apply_radio_discourse(
        lines,
        {"enabled": True, "greetings": ["のんばんは"], "split_framing_phrases": False},
    )

    assert len(result) == 2
    assert result[0].text == "のんばんは。"
    assert result[1].text == "のんばんは"
    assert result[0].end_time == pytest.approx(1.0)
    assert result[1].start_time == pytest.approx(1.0)


def test_apply_radio_discourse_greetings_adds_period_when_phrase_has_none():
    lines = [
        SubtitleLine(
            text="おはようございます今日も。",
            start_time=0.0,
            end_time=3.0,
            words=[
                TranscribedWord(word="おはよう", start_time=0.0, end_time=0.5),
                TranscribedWord(word="ございます", start_time=0.5, end_time=1.0),
                TranscribedWord(word="今日も。", start_time=1.0, end_time=3.0),
            ],
        ),
    ]

    result = apply_radio_discourse(
        lines,
        {
            "enabled": True,
            "greetings": ["おはようございます"],
            "split_framing_phrases": False,
        },
    )

    assert len(result) == 2
    assert result[0].text == "おはようございます。"
    assert result[1].text == "今日も。"


def test_apply_radio_discourse_greetings_attaches_comma_to_greeting_line():
    lines = [
        SubtitleLine(
            text="のんばんは、のんばんは",
            start_time=0.0,
            end_time=2.0,
            words=[
                TranscribedWord(word="のん", start_time=0.0, end_time=0.3),
                TranscribedWord(word="ばん", start_time=0.3, end_time=0.6),
                TranscribedWord(word="は、", start_time=0.6, end_time=1.0),
                TranscribedWord(word="のん", start_time=1.0, end_time=1.3),
                TranscribedWord(word="ばん", start_time=1.3, end_time=1.6),
                TranscribedWord(word="は", start_time=1.6, end_time=2.0),
            ],
        ),
    ]

    result = apply_radio_discourse(
        lines,
        {"enabled": True, "greetings": ["のんばんは"], "split_framing_phrases": False},
    )

    assert len(result) == 2
    assert result[0].text == "のんばんは、"
    assert result[1].text == "のんばんは"
    assert result[0].end_time == pytest.approx(1.0)
    assert result[1].start_time == pytest.approx(1.0)
