# Deferred AutoSub Planned Work

With the core MVP (Minimum Viable Product) now complete—featuring a fully connected `autosub run` pipeline that handles Transcription, Formatting, and Translation via the composable TOML `--profile` system—the following architectural plans have been mapped out for Phase 2 of the `autosub` toolchain.


## 2. Advanced Timing Rules
Currently, subtitle chunking relies purely on semantic pauses and punctuation. Phase 2 aims to introduce professional visual rules:
*   Limit text lines on screen to a maximum of 2.
*   Ensure there are no awkward invisible gaps between consecutive lines (snapping to nearest neighbor if under a threshold).
*   Implement visual keyframe snapping to ensure subtitles don't pop up during scene transitions.

## 3. Audio Extraction & Segmentation Pipeline
*   **Singing Filtering**: Intelligently detect and ignore singing sections in concert videos (e.g. leveraging `spleeter` or similar vocal detection tech), so the primary transcription module exclusively subtitiles the MC / spoken sections.

## 4. On-Screen Text OCR
*   **Visual Pipeline**: Implement optical character recognition (OCR) on the raw video footage.
*   **Integration**: Seamlessly interleave OCR-generated `.ass` lines (e.g., lower thirds, on-screen signs) with the speech-generated `.ass` lines, ensuring visual styles do not clash and timestamps overlap cleanly.

## 5. Web UI or Desktop GUI
*   Wrap the Typer CLI in a clean interface (e.g. a local React/Next.js dashboard) where users can easily drop videos, select profiles from a visual list, edit TOML/Markdown files directly in a rich text editor, and browse the staging bucket without using the terminal.
