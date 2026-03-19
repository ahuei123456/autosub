# Deferred AutoSub Planned Work

With the core MVP (Minimum Viable Product) now complete—featuring a fully connected `autosub run` pipeline that handles Transcription, Formatting, and Translation via the composable TOML `--profile` system—the following architectural plans have been mapped out for Phase 2 of the `autosub` toolchain.


## 2. Advanced Timing Rules
The single-speaker timing pipeline now supports minimum duration padding, gap snapping, optional keyframe-aware scene snapping, and automatic wrapping to keep subtitles within two visible lines.

Remaining work in this area:
*   Improve line breaking so semantic chunking and visual wrapping cooperate better on very dense speech.
*   Tune scene-aware snapping heuristics against real subtitle editing workflows in Aegisub.
*   Add profile presets for different content densities instead of relying on one global timing profile.

## 3. Audio Extraction & Segmentation Pipeline
*   **Singing Filtering**: Intelligently detect and ignore singing sections in concert videos (e.g. leveraging `spleeter` or similar vocal detection tech), so the primary transcription module exclusively subtitiles the MC / spoken sections.

## 4. On-Screen Text OCR
*   **Visual Pipeline**: Implement optical character recognition (OCR) on the raw video footage.
*   **Integration**: Seamlessly interleave OCR-generated `.ass` lines (e.g., lower thirds, on-screen signs) with the speech-generated `.ass` lines, ensuring visual styles do not clash and timestamps overlap cleanly.

## 5. Review UX
*   Add a lightweight review flow around `original.ass` so users can edit timings and line breaks in Aegisub before running translation.
