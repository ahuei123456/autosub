"""Resolve ffmpeg/ffprobe binary paths once at import time.

ffmpeg-python builds subprocess commands itself and does not always quote the
executable path correctly. On Windows, when ffmpeg lives in a path containing
spaces (e.g. ``C:\\Program Files\\ffmpeg\\bin\\ffmpeg.EXE``), the unquoted
command can fail. Passing the resolved path explicitly via ``cmd=`` makes
ffmpeg-python use it as-is and avoids the quoting bug.

Resolution order for each binary:
  1. ``FFMPEG_PATH`` / ``FFPROBE_PATH`` environment variable (explicit override)
  2. ``shutil.which()`` against the user's ``PATH``
  3. Common Windows install directories (no-op on POSIX)
  4. Bare name (``"ffmpeg"`` / ``"ffprobe"``) as a final fallback so we never
     raise where the prior behavior would have succeeded

Note: bare ``subprocess.Popen(["ffmpeg", ...])`` calls are NOT affected by the
underlying bug — execv-style invocation handles paths with spaces correctly via
PATH lookup. This indirection is only needed for ``ffmpeg-python``.
"""

import os
import shutil

_WINDOWS_FALLBACK_DIRS = (
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"C:\ffmpeg\bin",
)


def _resolve(name: str, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for directory in _WINDOWS_FALLBACK_DIRS:
        candidate = shutil.which(name, path=directory)
        if candidate:
            return candidate
    return name


FFMPEG_BIN = _resolve("ffmpeg", "FFMPEG_PATH")
FFPROBE_BIN = _resolve("ffprobe", "FFPROBE_PATH")
