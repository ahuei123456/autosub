# Deferred AutoSub Planned Work

This document holds the architectural plans for the next stages of the `autosub` toolchain. These features have been temporarily tabled to focus on refining the core `transcribe` module first.

## Goal Description
Design a modular, testable, and robust architecture for the `autosub` toolchain. Each step (transcription, formatting, translation) must be able to run independently or be chained together in an end-to-end workflow.

## Step 2: Subtitle Formatting (.ass generation)
The Formatting module's specific responsibility is to merge the individual words from Step 1 (`transcript.json`) and ultimately generate a `.ass` file.

*   Because many Python `.ass` libraries lack comprehensive documentation or functionality, a prerequisite step will be to select a wrapper library, parse its source code, and generate our own robust documentation (e.g., as an MCP server doc or markdown reference). This guarantees the AI agent can reliably generate correct formatting code against it.
*   **Input**: `transcript.json` (Array of Pydantic `TranscribedWord` objects)
*   **Output**: `original.ass`

## Step 3: Translation Module
The Translation module will take the fully formatted `.ass` file and translate the subtitle lines into the target language.

*   This step should use the Google Cloud Translation API.
*   It should be augmented with a Large Language Model (LLM) for context-aware and natural phrasing, improving upon the literal translations often provided by standard translation APIs.
*   **Input**: `original.ass`
*   **Output**: `translated.ass`
