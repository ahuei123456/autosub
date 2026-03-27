import re


def parse_timestamp(timestamp: str) -> float:
    """
    Parses a timestamp string in HH:MM:SS.mmm, MM:SS.mmm, or SS.mmm format
    into float seconds.
    """
    if not timestamp:
        return 0.0

    # Try to parse as seconds directly
    try:
        return float(timestamp)
    except ValueError:
        pass

    # HH:MM:SS.mmm or MM:SS.mmm
    parts = list(map(float, re.split("[:]", timestamp)))
    if len(parts) == 3:  # HH:MM:SS.mmm
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:  # MM:SS.mmm
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:
        return parts[0]
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
