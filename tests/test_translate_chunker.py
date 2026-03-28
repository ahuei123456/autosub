from autosub.pipeline.translate.chunker import (
    make_chunks,
    _find_corner_boundaries,
    _chunk_by_corners,
    MIN_CHUNK_SIZE,
)


def _texts(n):
    """Generate n dummy subtitle lines."""
    return [f"line {i}" for i in range(n)]


# --- _find_corner_boundaries ---


def test_find_boundaries_basic():
    texts = ["hello", "welcome to the corner", "more text", "another cue here"]
    boundaries = _find_corner_boundaries(texts, ["corner", "cue"])
    assert boundaries == [1, 3]


def test_find_boundaries_no_matches():
    texts = ["hello", "world", "foo"]
    assert _find_corner_boundaries(texts, ["zzz"]) == []


def test_find_boundaries_multiple_cues_same_line():
    texts = ["this has cue1 and cue2"]
    assert _find_corner_boundaries(texts, ["cue1", "cue2"]) == [0]


# --- _chunk_by_corners ---


def test_chunk_by_corners_basic():
    texts = _texts(30)
    # boundary at 15 splits into two chunks
    chunks, splits = _chunk_by_corners(texts, [15], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 15
    assert len(chunks[1]) == 15
    assert splits == set()


def test_chunk_by_corners_merges_tiny_segments():
    texts = _texts(30)
    # boundaries at 10 and 12 — gap of 2 is below MIN_CHUNK_SIZE, so 12 is dropped
    chunks, splits = _chunk_by_corners(texts, [10, 12], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 10
    assert len(chunks[1]) == 20  # 12 merged forward
    assert splits == set()


def test_chunk_by_corners_adjacent_boundaries():
    texts = _texts(50)
    # boundaries at 20 and 21 — only 1 line apart, second is dropped
    chunks, splits = _chunk_by_corners(texts, [20, 21], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 20
    assert len(chunks[1]) == 30
    assert splits == set()


def test_chunk_by_corners_subsplits_oversized():
    texts = _texts(100)
    # single boundary at 0 means one giant segment, max_chunk_size=30
    chunks, splits = _chunk_by_corners(texts, [], max_chunk_size=30)
    # no boundaries means filtered = [0], all_breaks = [0, 100]
    # segment of 100 gets sub-split at 30: 30+30+30+10
    assert len(chunks) == 4
    assert [len(c) for c in chunks] == [30, 30, 30, 10]
    assert splits == {30, 60, 90}


def test_chunk_by_corners_boundary_at_zero():
    texts = _texts(25)
    # boundary at 0 just reinforces the start, boundary at 15 splits
    chunks, splits = _chunk_by_corners(texts, [0, 15], max_chunk_size=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 15
    assert len(chunks[1]) == 10
    assert splits == set()


def test_chunk_by_corners_preserves_content():
    texts = ["a", "b", "c", "d", "e"] * 6  # 30 items
    chunks, splits = _chunk_by_corners(texts, [15], max_chunk_size=80)
    flat = [item for chunk in chunks for item in chunk]
    assert flat == texts


# --- make_chunks (integration) ---


def test_make_chunks_no_cues_fixed_chunking():
    texts = _texts(25)
    chunks, splits = make_chunks(texts, chunk_size=10)
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [10, 10, 5]
    assert splits == {10, 20}


def test_make_chunks_cues_no_matches_falls_back():
    texts = _texts(25)
    chunks, splits = make_chunks(texts, chunk_size=10, corner_cues=["zzz_no_match"])
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [10, 10, 5]
    assert splits == {10, 20}


def test_make_chunks_none_cues_fixed_chunking():
    texts = _texts(25)
    chunks, splits = make_chunks(texts, chunk_size=10, corner_cues=None)
    assert len(chunks) == 3
    assert splits == {10, 20}


def test_make_chunks_corner_aware():
    # Plant a cue at line 12
    texts = _texts(30)
    texts[12] = "さあ、カードイラストのコーナーです"
    chunks, splits = make_chunks(texts, chunk_size=80, corner_cues=["カードイラスト"])
    assert len(chunks) == 2
    assert len(chunks[0]) == 12
    assert len(chunks[1]) == 18
    # Corner-detected boundary — no artificial splits
    assert splits == set()


def test_make_chunks_preserves_all_lines():
    texts = _texts(50)
    texts[20] = "cue phrase here"
    chunks, splits = make_chunks(texts, chunk_size=80, corner_cues=["cue phrase"])
    flat = [item for chunk in chunks for item in chunk]
    assert flat == texts


# --- edge case: max_chunk_size < MIN_CHUNK_SIZE ---


def test_chunk_by_corners_small_max_chunk_size():
    texts = _texts(20)
    # boundary at 5, max_chunk_size=3 (smaller than MIN_CHUNK_SIZE)
    # min_size should clamp to 3, so boundary at 5 is kept
    chunks, splits = _chunk_by_corners(texts, [5], max_chunk_size=3)
    # chunk 0: lines 0-4 (5 lines) — oversized, sub-split at 3: [3, 2]
    # chunk 1: lines 5-19 (15 lines) — oversized, sub-split at 3: [3, 3, 3, 3, 3]
    assert sum(len(c) for c in chunks) == 20
    assert all(len(c) <= 3 for c in chunks)
    # Sub-splits are artificial: at indices 3 (from first segment) and 8,11,14,17 (from second)
    assert 3 in splits
    assert 8 in splits


def test_make_chunks_small_chunk_size_with_cues():
    texts = _texts(20)
    texts[8] = "transition cue"
    chunks, splits = make_chunks(texts, chunk_size=5, corner_cues=["transition cue"])
    # boundary at 8 is kept (8 - 0 = 8 >= min(10, 5) = 5)
    flat = [item for chunk in chunks for item in chunk]
    assert flat == texts
    # no chunk should exceed chunk_size=5
    assert all(len(c) <= 5 for c in chunks)


# --- splits-specific tests ---


def test_fixed_size_all_boundaries_in_splits():
    """All inter-chunk boundaries are artificial splits in fixed-size mode."""
    texts = _texts(40)
    chunks, splits = make_chunks(texts, chunk_size=10)
    assert splits == {10, 20, 30}


def test_corner_aware_no_artificial_splits():
    """Corner boundaries should NOT appear in splits."""
    texts = _texts(40)
    texts[20] = "corner cue here"
    chunks, splits = make_chunks(texts, chunk_size=80, corner_cues=["corner cue"])
    assert 20 not in splits
    assert splits == set()


def test_corner_aware_subsplits_in_splits():
    """Sub-splits of oversized segments ARE in splits."""
    texts = _texts(50)
    texts[10] = "corner cue here"
    # Corner at 10 → segments [0:10] (10 lines) and [10:50] (40 lines)
    # Second segment oversized with max=15 → sub-splits at 10+15=25, 10+30=40
    chunks, splits = make_chunks(texts, chunk_size=15, corner_cues=["corner cue"])
    assert 10 not in splits  # corner boundary, not artificial
    assert 25 in splits
    assert 40 in splits


def test_single_chunk_empty_splits():
    """A single chunk produces no splits."""
    texts = _texts(10)
    chunks, splits = make_chunks(texts, chunk_size=100)
    assert len(chunks) == 1
    assert splits == set()
