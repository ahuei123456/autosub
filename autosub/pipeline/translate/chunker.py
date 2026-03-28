import logging

logger = logging.getLogger(__name__)

MIN_CHUNK_SIZE = 10  # minimum lines per chunk for corner-aware splitting


def make_chunks(
    texts: list[str],
    chunk_size: int,
    corner_cues: list[str] | None = None,
) -> tuple[list[list[str]], set[int]]:
    """Split texts into chunks for translation.

    If corner_cues are provided, attempts to split at lines containing cue
    phrases so that segment transitions don't land at chunk boundaries.
    Falls back to fixed-size chunking when no cues or no matches are found.

    Returns (chunks, splits) where splits is a set of line indices in the
    original texts array where artificial (non-corner) boundaries occurred.
    """
    if corner_cues:
        boundaries = _find_corner_boundaries(texts, corner_cues)
        if boundaries:
            chunks, splits = _chunk_by_corners(texts, boundaries, chunk_size)
            chunk_sizes = [len(c) for c in chunks]
            logger.info(
                f"Corner-aware chunking: {len(boundaries)} boundaries found at lines "
                f"{boundaries}, producing {len(chunks)} chunks of sizes {chunk_sizes}"
            )
            return chunks, splits

    splits = {i for i in range(chunk_size, len(texts), chunk_size)}
    return [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)], splits


def _find_corner_boundaries(texts: list[str], cues: list[str]) -> list[int]:
    """Scan texts for corner cue phrases, return indices where segments start."""
    boundaries = []
    for i, text in enumerate(texts):
        if any(cue in text for cue in cues):
            boundaries.append(i)
    return boundaries


def _chunk_by_corners(
    texts: list[str], boundaries: list[int], max_chunk_size: int
) -> tuple[list[list[str]], set[int]]:
    """Split texts at corner boundaries, merging tiny segments and sub-splitting oversized ones.

    Boundaries that would create a chunk smaller than MIN_CHUNK_SIZE are
    dropped, merging the tiny segment into the next one.

    Returns (chunks, splits) where splits contains line indices of artificial
    sub-split boundaries (not corner-detected ones).
    """
    # Filter out boundaries that would create undersized chunks
    min_size = min(MIN_CHUNK_SIZE, max_chunk_size)
    filtered = [0]  # always start at 0
    for b in boundaries:
        if b - filtered[-1] >= min_size:
            filtered.append(b)
    # Add end boundary
    all_breaks = filtered + [len(texts)]

    chunks = []
    splits: set[int] = set()
    for start, end in zip(all_breaks, all_breaks[1:]):
        segment = texts[start:end]
        if len(segment) <= max_chunk_size:
            chunks.append(segment)
        else:
            logger.warning(
                f"  Segment at line {start} exceeds chunk_size "
                f"({len(segment)} > {max_chunk_size}), sub-splitting. "
                f"Consider adding more corners to the profile."
            )
            for j in range(0, len(segment), max_chunk_size):
                chunks.append(segment[j : j + max_chunk_size])
                if j > 0:
                    splits.add(start + j)
    return chunks, splits
