from autosub.core.schemas import SubtitleLine
from autosub.pipeline.format.timing import (
    apply_timing_rules,
    _apply_min_duration_padding,
    _apply_gap_snapping,
    _apply_micro_snapping,
    SegmentMS,
)


def test_min_duration_basic_padding():
    # Segment is 300ms, should pad 100ms on each side
    lines = [SubtitleLine(text="Short", start_time=0.2, end_time=0.5, speaker=None)]
    result = apply_timing_rules(lines, min_duration_ms=500)
    assert len(result) == 1
    assert result[0].start_time == 0.1
    assert result[0].end_time == 0.6


def test_min_duration_collision_and_merge():
    # L1: 200ms duration. Needs 300 padding. Boxed in on left by 0 limitation.
    # L2: 200ms duration.
    # They will collide and since there isn't enough space before they get boxed in, they merge.
    lines = [
        SubtitleLine(text="L1", start_time=0.1, end_time=0.3, speaker=None),
        SubtitleLine(text="L2", start_time=0.3, end_time=0.5, speaker=None),
    ]
    result = apply_timing_rules(lines, min_duration_ms=500, video_duration_ms=2000)

    # We expect L1 and L2 to merge into one 500ms line
    assert len(result) == 1
    assert result[0].text == "L1 L2"
    assert result[0].start_time == 0.0
    assert (
        result[0].end_time == 0.8
    )  # L2 will expand right to fulfill its padding requirement.
    # Actually L2 CAN expand rignt. L2 original end was 0.5. Needs 300ms padding.
    # It will expand to 0.8. Then L1 (which is 0.0 to 0.3) merges with L2 (0.3 to 0.8)
    # so the final should be 0.0 to 0.8! Let's check this via pytest


def test_gap_snapping_meet_in_middle():
    # 200ms gap, typical snap threshold is 250
    lines = [
        SubtitleLine(text="One", start_time=0.0, end_time=1.0, speaker=None),
        SubtitleLine(text="Two", start_time=1.2, end_time=2.2, speaker=None),
    ]
    result = apply_timing_rules(lines, snap_threshold_ms=250)
    # Gap is 200ms (1.0 to 1.2). Meets in middle at 1.1.
    assert result[0].end_time == 1.1
    assert result[1].start_time == 1.1


def test_keyframe_wall():
    # L1: 0.1 to 0.4 (300ms). Needs to expand 100ms on both sides.
    # Keyframe at 0.05. It can expand left up to 0.05 (50ms).
    # Then it shifts the burden right by 50ms, expanding right by 150ms.
    lines = [SubtitleLine(text="Key", start_time=0.1, end_time=0.4, speaker=None)]
    keyframes = [50]  # ms
    result = apply_timing_rules(lines, keyframes_ms=keyframes, min_duration_ms=500)
    assert result[0].start_time == 0.05
    assert result[0].end_time == 0.55


def test_micro_snapping():
    # L1: 0.0 to 1.0. Next keyframe is at 1.050 (50ms away).
    # It should micro-snap to 1.050 if threshold >= 50.
    lines = [SubtitleLine(text="Snap", start_time=0.0, end_time=1.0, speaker=None)]
    keyframes = [1050]
    result = apply_timing_rules(lines, keyframes_ms=keyframes, snap_threshold_ms=250)
    assert result[0].end_time == 1.05


def test_pass1_keyframe_hit_and_merge():
    # L1: 0.1 to 0.3 (200ms). Needs to pad 300ms.
    # L2: 0.33 to 0.5 (170ms).
    # Keyframe at 0.05. L1 can expand left up to 0.05 (50ms).
    # It must expand right by 250ms, but L2 is at 0.33. They will collide and merge.
    lines = [
        SubtitleLine(text="L1", start_time=0.1, end_time=0.3, speaker=None),
        SubtitleLine(text="L2", start_time=0.33, end_time=0.5, speaker=None),
    ]
    keyframes = [50]
    result = apply_timing_rules(
        lines, keyframes_ms=keyframes, min_duration_ms=500, video_duration_ms=2000
    )
    assert len(result) == 1
    assert result[0].text == "L1 L2"
    assert result[0].start_time == 0.05
    assert result[0].end_time >= 0.5


def test_min_duration_does_not_merge_listener_mail_into_host():
    lines = [
        SubtitleLine(
            text="のんばんは？",
            start_time=0.1,
            end_time=0.3,
            speaker=None,
            role="host",
        ),
        SubtitleLine(
            text="初メールです。",
            start_time=0.3,
            end_time=0.5,
            speaker=None,
            role="listener_mail",
        ),
    ]

    result = apply_timing_rules(lines, min_duration_ms=500, video_duration_ms=2000)

    assert len(result) == 2
    assert result[0].text == "のんばんは？"
    assert result[1].text == "初メールです。"
    assert result[0].role == "host"
    assert result[1].role == "listener_mail"


def test_min_duration_can_still_merge_when_roles_match():
    lines = [
        SubtitleLine(
            text="L1",
            start_time=0.1,
            end_time=0.3,
            speaker=None,
            role="host",
        ),
        SubtitleLine(
            text="L2",
            start_time=0.3,
            end_time=0.5,
            speaker=None,
            role="host",
        ),
    ]

    result = apply_timing_rules(lines, min_duration_ms=500, video_duration_ms=2000)

    assert len(result) == 1
    assert result[0].text == "L1 L2"
    assert result[0].role == "host"


def test_pass1_proportional_gap_division():
    # L1: 0.5 to 0.7 (200ms). Needs 300ms padding.
    # Gap: 0.7 to 0.9 (200ms).
    # L2: 0.9 to 1.1 (200ms). Needs 300ms padding.
    # Both need 300ms padding. Initial request: 150ms left, 150ms right for each.
    # Gap is 200 < 300 (requested total into gap). They split the 200ms gap 50/50 (100ms each).
    # L1 then shifts its remaining 50ms burden left. => 200ms left, 100ms right.
    # L2 shifts its remaining 50ms burden right. => 100ms left, 200ms right.
    lines = [
        SubtitleLine(text="L1", start_time=0.5, end_time=0.7, speaker=None),
        SubtitleLine(text="L2", start_time=0.9, end_time=1.1, speaker=None),
    ]
    result = apply_timing_rules(lines, min_duration_ms=500, video_duration_ms=2000)
    assert len(result) == 2
    # L1 goes from 0.5 - 0.2 = 0.3. End goes from 0.7 + 0.1 = 0.8.
    assert result[0].start_time == 0.3
    assert result[0].end_time == 0.8
    # L2 goes from 0.9 - 0.1 = 0.8. End goes from 1.1 + 0.2 = 1.3.
    assert result[1].start_time == 0.8
    assert result[1].end_time == 1.3


def test_pass1_video_end_boundary_blocking():
    # Final line: 1.8 to 1.9 (100ms). Needs 400ms padding.
    # Video ends at 2.0. Can expand right by 100ms.
    # Total so far: 1.8 to 2.0 (200ms). Needs 300ms more on the left.
    # Expands left to 1.5. Total: 1.5 to 2.0 (500ms).
    # But what if left is blocked by another line?
    lines = [
        SubtitleLine(text="L1", start_time=1.0, end_time=1.7, speaker=None),
        SubtitleLine(text="L2", start_time=1.8, end_time=1.9, speaker=None),
    ]
    result = apply_timing_rules(lines, min_duration_ms=500, video_duration_ms=2000)
    assert len(result) == 2
    assert result[1].end_time == 2.0
    # Gap 1.7 to 1.8 (100ms). L2 takes it. Left is 1.6 (but L1 ends at 1.7 so it can only go back to 1.7)
    # So L2 cannot reach 500ms. It stays short because it's the final line.
    assert result[1].start_time == 1.7
    assert result[1].end_time == 2.0


def test_pass2_small_gap_with_single_keyframe():
    # Gap 1.0 to 1.2 (200ms) < 250ms threshold.
    # Keyframe at 1.05.
    lines = [
        SubtitleLine(text="One", start_time=0.0, end_time=1.0, speaker=None),
        SubtitleLine(text="Two", start_time=1.2, end_time=2.2, speaker=None),
    ]
    keyframes = [1050]
    result = apply_timing_rules(lines, keyframes_ms=keyframes, snap_threshold_ms=250)
    assert result[0].end_time == 1.05
    assert result[1].start_time == 1.05


def test_pass2_gap_with_multiple_keyframes():
    # Gap 1.0 to 1.4 (400ms). Between 250 and 500 threshold.
    # Keyframes at 1.1 and 1.3.
    lines = [
        SubtitleLine(text="One", start_time=0.0, end_time=1.0, speaker=None),
        SubtitleLine(text="Two", start_time=1.4, end_time=2.4, speaker=None),
    ]
    keyframes = [1100, 1300]
    result = apply_timing_rules(
        lines, keyframes_ms=keyframes, conditional_snap_threshold_ms=500
    )
    assert result[0].end_time == 1.1
    assert result[1].start_time == 1.3


def test_pass2_conditional_snapping_threshold():
    # Gap 1.0 to 1.4 (400ms). No keyframes.
    # Meets in the middle at 1.2.
    lines = [
        SubtitleLine(text="One", start_time=0.0, end_time=1.0, speaker=None),
        SubtitleLine(text="Two", start_time=1.4, end_time=2.4, speaker=None),
    ]
    result = apply_timing_rules(lines, conditional_snap_threshold_ms=500)
    assert result[0].end_time == 1.2
    assert result[1].start_time == 1.2


def test_pass1_min_duration_isolated():
    lines = [SubtitleLine(text="Short", start_time=0.2, end_time=0.5, speaker=None)]
    segments = [SegmentMS(line) for line in lines]
    result = _apply_min_duration_padding(
        segments, keyframes=[], video_duration_ms=None, min_duration_ms=500
    )
    assert len(result) == 1
    assert result[0].start_ms == 100
    assert result[0].end_ms == 600


def test_pass2_gap_snapping_isolated():
    lines = [
        SubtitleLine(text="One", start_time=0.0, end_time=1.0, speaker=None),
        SubtitleLine(text="Two", start_time=1.2, end_time=2.2, speaker=None),
    ]
    segments = [SegmentMS(line) for line in lines]
    result = _apply_gap_snapping(
        segments,
        keyframes=[],
        snap_threshold_ms=250,
        conditional_snap_threshold_ms=500,
    )
    assert result[0].end_ms == 1100
    assert result[1].start_ms == 1100


def test_pass3_micro_snapping_isolated():
    lines = [SubtitleLine(text="Snap", start_time=0.0, end_time=1.0, speaker=None)]
    segments = [SegmentMS(line) for line in lines]
    result = _apply_micro_snapping(
        segments, keyframes=[1050], micro_snap_threshold=250, video_duration_ms=None
    )
    assert result[0].end_ms == 1050
