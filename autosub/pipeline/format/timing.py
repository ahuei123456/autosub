from typing import List, Optional

from autosub.core.schemas import SubtitleLine


class SegmentMS:
    """Helper wrapper to manipulate timestamps in raw milliseconds to avoid float drift."""

    def __init__(self, line: SubtitleLine):
        self.text = line.text
        self.speaker = line.speaker
        self.role = line.role
        self.corner = line.corner
        self.start_ms = int(round(line.start_time * 1000))
        self.end_ms = int(round(line.end_time * 1000))

    def to_subtitle_line(self) -> SubtitleLine:
        return SubtitleLine(
            text=self.text,
            start_time=self.start_ms / 1000.0,
            end_time=self.end_ms / 1000.0,
            speaker=self.speaker,
            role=self.role,
            corner=self.corner,
        )


class ProposedExtension:
    def __init__(self):
        self.lead_in = 0
        self.lead_out = 0


def _get_prev_keyframe(time_ms: int, keyframes: List[int]) -> Optional[int]:
    """Finds the closest keyframe before or at the given time."""
    for k in reversed(keyframes):
        if k <= time_ms:
            return k
    return None


def _get_next_keyframe(time_ms: int, keyframes: List[int]) -> Optional[int]:
    """Finds the closest keyframe after or at the given time."""
    for k in keyframes:
        if k >= time_ms:
            return k
    return None


def _apply_min_duration_padding(
    segments: List[SegmentMS],
    keyframes: List[int],
    video_duration_ms: Optional[int],
    min_duration_ms: int,
) -> List[SegmentMS]:
    """Pass 1: Minimum Duration (Analyze and Commit)"""
    extensions = [ProposedExtension() for _ in segments]

    # Phase A: Analyze
    for i, seg in enumerate(segments):
        duration = seg.end_ms - seg.start_ms
        if duration < min_duration_ms:
            shortfall = min_duration_ms - duration
            lead_in = shortfall // 2
            lead_out = shortfall - lead_in
            extensions[i].lead_in = lead_in
            extensions[i].lead_out = lead_out

    # Distribute gap spaces fairly
    for i in range(len(segments) - 1):
        ext_left = extensions[i].lead_out
        ext_right = extensions[i + 1].lead_in
        gap = segments[i + 1].start_ms - segments[i].end_ms
        gap = max(0, gap)

        # Keyframe bounds within this gap
        gap_kf = [
            k for k in keyframes if segments[i].end_ms < k < segments[i + 1].start_ms
        ]

        if gap_kf:
            max_left = gap_kf[0] - segments[i].end_ms
            max_right = segments[i + 1].start_ms - gap_kf[-1]
            extensions[i].lead_out = min(max_left, extensions[i].lead_out)
            extensions[i + 1].lead_in = min(max_right, extensions[i + 1].lead_in)
        else:
            if ext_left + ext_right > gap:
                # Proportional 50/50 division
                if ext_left > 0 and ext_right > 0:
                    extensions[i].lead_out = gap // 2
                    extensions[i + 1].lead_in = gap - (gap // 2)
                elif ext_left > 0:
                    extensions[i].lead_out = gap
                elif ext_right > 0:
                    extensions[i + 1].lead_in = gap

    # Check boundaries against video start/end
    for i, seg in enumerate(segments):
        max_lead_in = seg.start_ms
        kf_before = _get_prev_keyframe(seg.start_ms, keyframes)
        if kf_before is not None:
            max_lead_in = min(max_lead_in, seg.start_ms - kf_before)
        if i == 0:
            extensions[i].lead_in = min(extensions[i].lead_in, max_lead_in)

        max_lead_out = float("inf")
        if video_duration_ms is not None:
            max_lead_out = video_duration_ms - seg.end_ms
        kf_after = _get_next_keyframe(seg.end_ms, keyframes)
        if kf_after is not None:
            max_lead_out = min(max_lead_out, kf_after - seg.end_ms)
        if i == len(segments) - 1:
            if max_lead_out != float("inf"):
                extensions[i].lead_out = min(extensions[i].lead_out, int(max_lead_out))

    # Commit extensions
    for i, seg in enumerate(segments):
        duration = seg.end_ms - seg.start_ms
        if duration >= min_duration_ms:
            continue

        shortfall = min_duration_ms - duration
        ext = extensions[i]

        actual_lead_in = ext.lead_in
        actual_lead_out = ext.lead_out

        # Recalculate segment specific bounds to avoid python loop scope bleed
        cur_max_lead_in = seg.start_ms
        kf_before = _get_prev_keyframe(seg.start_ms, keyframes)
        if kf_before is not None:
            cur_max_lead_in = min(cur_max_lead_in, seg.start_ms - kf_before)

        cur_max_lead_out = float("inf")
        if video_duration_ms is not None:
            cur_max_lead_out = video_duration_ms - seg.end_ms
        kf_after = _get_next_keyframe(seg.end_ms, keyframes)
        if kf_after is not None:
            cur_max_lead_out = min(cur_max_lead_out, kf_after - seg.end_ms)

        total_padding = actual_lead_in + actual_lead_out
        if total_padding < shortfall:
            # Shift the burden right
            if i < len(segments) - 1:
                right_gap = segments[i + 1].start_ms - segments[i].end_ms
                kf_gap = [
                    k
                    for k in keyframes
                    if segments[i].end_ms < k < segments[i + 1].start_ms
                ]
                if kf_gap:
                    right_gap = kf_gap[0] - segments[i].end_ms
                available_right = right_gap - extensions[i + 1].lead_in
            else:
                available_right = cur_max_lead_out

            if available_right > actual_lead_out:
                additional = min(
                    available_right - actual_lead_out, shortfall - total_padding
                )
                if additional != float("inf"):
                    actual_lead_out += int(additional)

            # Shift the burden left
            if i > 0:
                left_gap = segments[i].start_ms - segments[i - 1].end_ms
                kf_gap = [
                    k
                    for k in keyframes
                    if segments[i - 1].end_ms < k < segments[i].start_ms
                ]
                if kf_gap:
                    left_gap = segments[i].start_ms - kf_gap[-1]
                available_left = left_gap - extensions[i - 1].lead_out
            else:
                available_left = cur_max_lead_in

            total_padding = actual_lead_in + actual_lead_out
            if available_left > actual_lead_in and total_padding < shortfall:
                additional = min(
                    available_left - actual_lead_in, shortfall - total_padding
                )
                if additional != float("inf"):
                    actual_lead_in += int(additional)

        # Commit
        seg.start_ms -= int(actual_lead_in)
        seg.end_ms += int(actual_lead_out)
        extensions[i].lead_in = int(actual_lead_in)
        extensions[i].lead_out = int(actual_lead_out)

    # Merging
    merged_segments = []
    skip_next = False
    for i in range(len(segments)):
        if skip_next:
            skip_next = False
            continue

        seg = segments[i]
        duration = seg.end_ms - seg.start_ms
        if duration < min_duration_ms:
            if i < len(segments) - 1:
                # Merge with the next segment
                next_seg = segments[i + 1]
                # Combine text
                seg.text = f"{seg.text} {next_seg.text}".strip()
                seg.end_ms = next_seg.end_ms
                skip_next = True
            else:
                pass  # Final Segment edge case
        merged_segments.append(seg)
    return merged_segments


def _apply_gap_snapping(
    segments: List[SegmentMS],
    keyframes: List[int],
    snap_threshold_ms: int,
    conditional_snap_threshold_ms: int,
) -> List[SegmentMS]:
    """Pass 2: Gaps"""
    for i in range(len(segments) - 1):
        prev_seg = segments[i]
        next_seg = segments[i + 1]

        gap = next_seg.start_ms - prev_seg.end_ms
        if gap <= 0:
            continue

        if gap < snap_threshold_ms:
            kfs_in_gap = [
                k for k in keyframes if prev_seg.end_ms < k < next_seg.start_ms
            ]
            if kfs_in_gap:
                # Small Gap with Keyframe
                prev_seg.end_ms = kfs_in_gap[0]
                next_seg.start_ms = kfs_in_gap[0]
            else:
                # Meeting in the middle
                half_gap = gap // 2
                prev_seg.end_ms += half_gap
                next_seg.start_ms -= gap - half_gap
        elif snap_threshold_ms <= gap < conditional_snap_threshold_ms:
            kfs_in_gap = [
                k for k in keyframes if prev_seg.end_ms < k < next_seg.start_ms
            ]
            if kfs_in_gap:
                # Multiple Keyframes in Gap
                prev_seg.end_ms = kfs_in_gap[0]
                next_seg.start_ms = kfs_in_gap[-1]
            else:
                # Conditional Gap (no keyframes) - snap standard
                half_gap = gap // 2
                prev_seg.end_ms += half_gap
                next_seg.start_ms -= gap - half_gap
    return segments


def _apply_micro_snapping(
    segments: List[SegmentMS],
    keyframes: List[int],
    micro_snap_threshold: int,
    video_duration_ms: Optional[int],
) -> List[SegmentMS]:
    """Pass 3: Micro-Snapping"""
    for i, seg in enumerate(segments):
        start_kf = _get_prev_keyframe(seg.start_ms, keyframes)
        if start_kf is not None:
            dist = seg.start_ms - start_kf
            if 0 < dist <= micro_snap_threshold:
                prev_end = segments[i - 1].end_ms if i > 0 else 0
                if start_kf >= prev_end:
                    seg.start_ms = start_kf

        end_kf = _get_next_keyframe(seg.end_ms, keyframes)
        if end_kf is not None:
            dist = end_kf - seg.end_ms
            if 0 < dist <= micro_snap_threshold:
                next_start = (
                    segments[i + 1].start_ms
                    if i < len(segments) - 1
                    else (video_duration_ms or float("inf"))
                )
                if end_kf <= next_start:
                    seg.end_ms = end_kf
    return segments


def apply_timing_rules(
    lines: List[SubtitleLine],
    keyframes_ms: Optional[List[int]] = None,
    video_duration_ms: Optional[int] = None,
    min_duration_ms: int = 500,
    snap_threshold_ms: int = 250,
    conditional_snap_threshold_ms: int = 500,
) -> List[SubtitleLine]:
    """Applies advanced timing rules to subtitle lines."""

    if not lines:
        return []

    keyframes = sorted(keyframes_ms) if keyframes_ms else []
    segments = [SegmentMS(line) for line in lines]

    segments = _apply_min_duration_padding(
        segments, keyframes, video_duration_ms, min_duration_ms
    )
    segments = _apply_gap_snapping(
        segments, keyframes, snap_threshold_ms, conditional_snap_threshold_ms
    )
    segments = _apply_micro_snapping(
        segments, keyframes, snap_threshold_ms, video_duration_ms
    )

    # Final Bounds Check
    for seg in segments:
        if seg.start_ms < 0:
            seg.start_ms = 0
        if video_duration_ms is not None and seg.end_ms > video_duration_ms:
            seg.end_ms = video_duration_ms

    return [seg.to_subtitle_line() for seg in segments]
