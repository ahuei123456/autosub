# autosub

Automatic Japanese subtitle generation and translation pipeline for speech-heavy video and audio.

## Current Pipeline

`autosub` currently runs a four-stage CLI pipeline:

1. **Transcribe**: Extract audio, send it to Google Cloud Speech-to-Text (`chirp_2`), and write a word-timed `transcript.json`.
2. **Format**: Chunk words into subtitle lines, optionally apply discourse-aware radio segmentation, apply timing and optional keyframe snapping, and write `original.ass`.
3. **Translate**: Translate subtitle events with either Vertex AI (`gemini-3-flash-preview`) or Cloud Translation v3, then write `translated.ass`.
4. **Postprocess**: Apply profile-driven editorial cleanup to the translated `.ass` file. The built-in `run` command includes this step automatically.

```mermaid
graph TD
    A[Video or Audio Input] --> B[Transcribe<br/>Google Speech-to-Text chirp_2]
    B --> C[Format<br/>Chunking + Timing + Optional Keyframe Snapping]
    C --> D[Translate<br/>Vertex Gemini 3 Flash or Cloud Translation v3]
    D --> E[Postprocess<br/>Profile Extensions]
    E --> F[Final translated.ass]
```

## Current Capabilities

- Word-level transcript timing from Google Speech-to-Text.
- Automatic short-audio local transcription and long-audio GCS batch transcription.
- Subtitle timing cleanup with minimum-duration padding, gap snapping, and optional keyframe alignment.
- Radio-show discourse extensions that can split listener mail framing and label subtitle roles.
- Optional bilingual output with original Japanese stacked above the translation.
- Profile inheritance for prompts, vocabulary, timing, and extensions.

## Current Limits

- The CLI is still documented and exposed as a **single-speaker** pipeline.
- The transcript and formatter can preserve `speaker` labels if they are already present in `transcript.json`, and `.ass` generation will create per-speaker styles, but diarization is not wired through the transcription commands yet.
- The formatter does **not** currently insert ASS line breaks (`\N`). Layout helpers exist in the codebase, but profile options such as `max_line_width` and `max_lines_per_subtitle` are not currently consumed by the CLI pipeline.

## Prerequisites

1. Python 3.12+
2. `uv`
3. FFmpeg available on `PATH`
4. Credentials for the services you plan to use:
   - Google Cloud for transcription, Cloud Translation v3, or `google-vertex`
   - `ANTHROPIC_API_KEY` for direct Anthropic translation or classification
   - `OPENAI_API_KEY` for direct OpenAI translation or classification
5. Google Cloud credentials when using Google-backed features:
   - `GOOGLE_APPLICATION_CREDENTIALS`
   - `GOOGLE_CLOUD_PROJECT`
   - `AUTOSUB_GCS_BUCKET` for audio longer than about 60 seconds
6. Optional: `SCXvid` if you want automatic keyframe extraction for scene-aware timing

## Installation

```powershell
git clone https://github.com/yourusername/autosub.git
Set-Location autosub
uv sync
```

## Configuration

Create a `.env` file in the repo root:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
AUTOSUB_GCS_BUCKET=your-staging-bucket
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
```

Notes:

- `ANTHROPIC_API_KEY` is only needed for `--llm-provider anthropic`.
- `OPENAI_API_KEY` is only needed for `--llm-provider openai`.
- `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, and `AUTOSUB_GCS_BUCKET` are only needed for Google-backed stages.
- Long-audio transcription still requires Google Cloud Storage even if translation uses Anthropic.

## Quick Start

Run the full pipeline:

```powershell
uv run autosub run .\video.mp4 --profile suzuhara_nozomi
```

For bilingual output:

```powershell
uv run autosub run .\video.mp4 --profile suzuhara_nozomi --bilingual
```

By default, `run` writes these files next to the input media:

- `transcript.json`
- `original.ass`
- `original.llm_trace.jsonl` when radio-discourse classification uses an LLM-backed engine
- `translated.ass`
- `translated.llm_trace.jsonl` when translation uses the Vertex LLM engine

If keyframe extraction is enabled and succeeds, it also writes `<input-stem>_keyframes.log`.

## Running Stages Individually

Transcribe:

```powershell
uv run autosub transcribe .\video.mp4 `
  --out .\transcript.json `
  --profile suzuhara_nozomi
```

Format with an existing keyframe log:

```powershell
uv run autosub format .\transcript.json `
  --out .\original.ass `
  --keyframes .\video_keyframes.log `
  --fps 23.976 `
  --profile suzuhara_nozomi
```

Translate with Anthropic:

```powershell
uv run autosub translate .\original.ass `
  --out .\translated.ass `
  --profile suzuhara_nozomi `
  --model claude-haiku-4-5 `
  --vertex-reasoning-effort low `
  --bilingual
```

Translate with Anthropic Sonnet 4.6:

```powershell
uv run autosub translate .\original.ass `
  --out .\translated.ass `
  --profile suzuhara_nozomi `
  --model claude-sonnet-4-6 `
  --vertex-reasoning-effort low `
  --chunk-size 20 `
  --bilingual
```

Translate with OpenAI:

```powershell
uv run autosub translate .\original.ass `
  --out .\translated.ass `
  --profile suzuhara_nozomi `
  --model gpt-5-mini `
  --vertex-reasoning-effort low `
  --bilingual
```

Postprocess a translated file explicitly:

```powershell
uv run autosub postprocess .\translated.ass `
  --profile suzuhara_nozomi `
  --bilingual
```

## Default CLI Config

`autosub` now auto-loads `.\config.toml` when that file exists.

Precedence is:

- explicit CLI flags
- `config.toml`
- built-in command defaults

You can override the path with `--config .\some-other.toml` or disable config loading for one run with `--no-config`.

The repo root now includes a template [`config.toml`](./config.toml) with one section per pipeline stage command:

- `[transcribe]`
- `[format]`
- `[translate]`
- `[postprocess]`

`autosub run` derives its defaults from those stage sections automatically, so it does not need its own config block in the normal case.

Use `config.toml` for command-line defaults you want every run to inherit, especially:

- `profile`
- `model`
- `llm_provider`
- `reasoning_effort`
- `bilingual`
- `chunk_size`
- `start` / `end`

For `autosub run`, the effective defaults are combined from the stage sections:

- transcription flags such as `language`, `vocab`, `profile`, `start`, and `end` come from `[transcribe]`
- formatting flags such as `keyframes` and `profile` come from `[format]`
- translation flags such as `prompt`, `target`, `source`, `model`, `llm_provider`, `reasoning_effort`, `bilingual`, and `chunk_size` come from `[translate]`
- if `[translate]` does not set `bilingual`, `run` falls back to `[postprocess].bilingual`

If you need a run-only override such as `out_dir` or `extract_keyframes`, an optional `[run]` section is still supported, but the default template leaves it out on purpose.

Keep reusable content in `profiles/*.toml` instead:

- prompt text
- vocabulary lists
- glossary entries
- timing rules
- replacements
- extension settings

## Command Reference

### `autosub transcribe`

- `--out`, `-o`: Output transcript path. Default: `transcript.json`
- `--language`, `-l`: Speech recognition language code. Default: `ja-JP`
- `--vocab`, `-v`: Additional speech adaptation hints. Can be passed multiple times.
- `--profile`: Loads profile vocabulary.

Behavior notes:

- Audio shorter than about 60 seconds is sent directly to the API.
- Longer audio is uploaded to GCS first and transcribed as a batch job.

### `autosub format`

- `--out`: Output `.ass` path. Default: `original.ass` in the transcript directory
- `--keyframes`: Path to an Aegisub keyframe log
- `--fps`: Required when `--keyframes` is used
- `--profile`: Loads `[timing]` and `[extensions]`

Behavior notes:

- Chunking is punctuation- and pause-aware.
- If speaker labels are already present in the transcript JSON, chunking is done per speaker and the generated `.ass` file gets one style per speaker.
- The `radio_discourse` extension runs here when enabled.

### `autosub translate`

- `--out`: Output `.ass` path. Default: `translated.ass`
- `--engine`, `-e`: `vertex` or `cloud-v3`
- `--model`: Preferred LLM selector for the `vertex` engine. Infers provider automatically for Gemini, Claude, and OpenAI model names
- `--llm-provider`: `google-vertex`, `anthropic`, or `openai` for the `vertex` engine
- `--prompt`, `-p`: Extra translation guidance appended after profile prompts
- `--profile`: Loads prompt text from the selected profile
- `--target`: Target language code. Default: `en`
- `--source`: Source language code. Default: `ja`
- `--llm-model` / `--vertex-model`: Override the LLM model name. Defaults to `gemini-3-flash-preview` for `google-vertex`, `claude-haiku-4-5` for `anthropic`, and `gpt-5-mini` for `openai`
- `--llm-location` / `--vertex-location`: Override the LLM location or region
- `--vertex-reasoning-effort`: Provider-agnostic reasoning effort for LLM-backed translation. Current support varies by provider and model family and can include `off`, `minimal`, `low`, `medium`, `high`
- `--vertex-reasoning-budget`: Optional token-budget override for provider-specific reasoning controls
- `--vertex-reasoning-dynamic` / `--no-vertex-reasoning-dynamic`: Request dynamic reasoning budget on supported providers and model families
- `--bilingual` / `--replace`: Stack Japanese above the translation, or replace text entirely
- `--chunk-size`: Number of subtitle lines per translation chunk. Use `0` to disable chunking. Default: `0`

Behavior notes:

- `vertex` uses the structured LLM path. The default provider is Vertex AI with `gemini-3-flash-preview`, and direct Anthropic is also supported with `--llm-provider anthropic`.
- `vertex` also supports direct OpenAI with `--llm-provider openai`.
- `cloud-v3` uses Google Cloud Translation v3 and ignores custom prompt text.
- `--model gemini-*` infers `google-vertex`, `--model claude-*` infers `anthropic`, and `--model gpt-*` or `o*` infers `openai`.
- `--model` cannot be combined with `--engine cloud-v3`.

Anthropic notes:

- Supported direct Anthropic models include `claude-haiku-4-5`, `claude-sonnet-4-6`, and `claude-opus-4-6`.
- When `--llm-provider anthropic` is selected and `--llm-model` is omitted, the default model is `claude-haiku-4-5`.
- For longer or stricter JSON-heavy translation jobs, `claude-sonnet-4-6` is usually more reliable than Haiku.
- `--llm-location` is ignored for direct Anthropic requests.
- Direct Anthropic uses the same `--vertex-reasoning-effort` flag for now because the CLI predates multi-provider support.

Current Anthropic reasoning defaults in this repo:

- `minimal`: thinking budget `2048`, `max_tokens 16384`
- `low`: thinking budget `4096`, `max_tokens 16384`
- `medium`: thinking budget `16384`, `max_tokens 32768`
- `high`: thinking budget `32768`, `max_tokens 65536`

OpenAI notes:

- Direct OpenAI currently defaults to `gpt-5-mini` when `--llm-provider openai` is selected and `--llm-model` is omitted.
- `--llm-location` is ignored for direct OpenAI requests.
- OpenAI uses the same `--vertex-reasoning-effort` flag for now because the CLI predates multi-provider support.
- `--vertex-reasoning-budget` is currently ignored for OpenAI. Use reasoning effort only.

### `autosub postprocess`

- `--out`: Output `.ass` path. Default: overwrite the input file
- `--profile`: Loads `[extensions]`
- `--bilingual` / `--replace`: Tells postprocess whether it is operating on stacked bilingual text or translated-only text

Behavior notes:

- Postprocessing only changes files when an enabled extension actually makes edits.
- The built-in `run` command postprocesses `translated.ass` in place.

### `autosub run`

`run` combines the full pipeline above and keeps the common end-to-end options:

- `--out-dir`
- `--language`
- `--profile`
- `--vocab`
- `--prompt`
- `--target`
- `--source`
- `--vertex-reasoning-effort`
- `--llm-provider`
- `--bilingual` / `--replace`
- `--keyframes`
- `--extract-keyframes` / `--no-extract-keyframes`
- `--start`
- `--end`
- `--chunk-size`

Behavior notes:

- `run` defaults to the Vertex AI translation path, but you can switch to direct Anthropic with `--llm-provider anthropic`.
- If you need `cloud-v3` or advanced LLM overrides such as model, location, or dynamic reasoning settings, run the stages separately and use `autosub translate`.

Example:

```powershell
uv run autosub run .\video.mp4 `
  --profile suzuhara_nozomi `
  --model claude-haiku-4-5 `
  --vertex-reasoning-effort low `
  --bilingual
```

## Unified Profile Format

Profiles live in [`profiles`](./profiles) and are loaded by name with `--profile <name>`.

Example:

```toml
extends = ["solo_seiyuu_radio"]

prompt = "prompts/suzuhara_nozomi.md"
vocab = [
    "鈴原希実",
    "のんちゃん",
]

[timing]
min_duration_ms = 700
snap_threshold_ms = 250
conditional_snap_threshold_ms = 500

[extensions.radio_discourse]
enabled = true
engine = "hybrid"
provider = "anthropic"
model = "claude-haiku-4-5"
reasoning_effort = "low"
scope = "full_script"
window_size = 10
window_overlap = 3
split_framing_phrases = true
label_roles = true
```

### Profile Keys

- `extends`: List of base profile names. Base profiles are loaded first.
- `prompt`: Either inline text or a path ending in `.md` or `.txt`. File contents are loaded into the translation prompt.
- `vocab`: List of speech adaptation hints. Inherited lists are appended.
- `[timing]`: Timing options for the formatter.
- `[extensions]`: Nested extension configuration shared by formatting and postprocessing.

### Prompt and Vocab Merge Rules

- Prompt fragments are concatenated in inheritance order: base profile first, child profile after that, then CLI `--prompt` last.
- Vocabulary entries are appended in the same order: base profile, child profile, then CLI `--vocab`.

### Timing Options Currently Wired Up

These keys are currently consumed by the formatter:

- `min_duration_ms`
- `snap_threshold_ms`
- `conditional_snap_threshold_ms`

No layout-related profile keys are currently wired into the CLI formatter.

## `radio_discourse` Extension

The built-in `radio_discourse` extension is designed for solo seiyuu radio or listener-mail style content.

During **formatting**, it can:

- split trailing framing phrases such as `といただきました。` into their own subtitle lines
- classify lines as `host`, `listener_mail`, or `host_meta`
- store the resolved role in the ASS event `Name` field

During **postprocessing**, it can:

- wrap translated `listener_mail` lines in quotation marks
- in bilingual mode, quote only the translated bottom line and leave the original Japanese untouched

Supported options:

- `enabled`: Turn the extension on
- `engine`: `rules`, `vertex`, or `hybrid`
- `provider`: `google-vertex` or `anthropic` for the LLM-backed modes
- `provider`: `google-vertex`, `anthropic`, or `openai` for the LLM-backed modes
- `model`: LLM model name for `vertex` or `hybrid`
- `reasoning_effort`: Provider-agnostic reasoning effort for LLM-backed classification. Current support varies by provider and model family and can include `off`, `minimal`, `low`, `medium`, `high`
- `reasoning_budget_tokens`: Optional token-budget override for provider-specific reasoning controls
- `reasoning_dynamic`: Request dynamic reasoning budget when the selected provider supports it
- `location`: LLM location or region. Default: `global`
- `scope`: `full_script` or windowed classification
- `window_size`: Window size for non-`full_script` classification
- `window_overlap`: Window overlap for non-`full_script` classification
- `split_framing_phrases`: Split host framing suffixes into separate lines before classification
- `label_roles`: Persist the resolved role onto subtitle events

`hybrid` uses rule-based labels first and falls back to them if LLM classification fails.

Anthropic-backed `radio_discourse` example:

```toml
[extensions.radio_discourse]
enabled = true
engine = "hybrid"
provider = "anthropic"
model = "claude-haiku-4-5"
reasoning_effort = "low"
scope = "full_script"
split_framing_phrases = true
label_roles = true
```

OpenAI-backed `radio_discourse` example:

```toml
[extensions.radio_discourse]
enabled = true
engine = "hybrid"
provider = "openai"
model = "gpt-5-mini"
reasoning_effort = "low"
scope = "full_script"
split_framing_phrases = true
label_roles = true
```
