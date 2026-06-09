# CLAUDE.md

Automatic Japanese subtitle generation and translation pipeline. Fork of `ahuei123456/autosub`, hosted at `mting314/autosub`.

## Quick Reference

- **Language**: Python 3.12+, managed with `uv`
- **CLI framework**: Typer, entry point `autosub.cli:app`
- **Run**: `uv run autosub <command>`
- **Tests**: `uv run pytest` (or `uv run pytest tests/<file>` for a subset)
- **Lint**: `uv run ruff check`

## Project Layout

```
autosub/
├── cli.py                     # Typer CLI commands
├── core/
│   ├── llm/                   # LLM provider abstraction (Vertex, Anthropic, OpenAI, OpenRouter)
│   ├── profile.py             # TOML profile loader with inheritance (extends)
│   ├── speaker_map.py         # Speaker map TOML parsing and assignment
│   └── ...
├── extensions/
│   ├── corners/               # Corner (program segment) detection
│   └── radio_discourse/       # Radio show discourse classification
├── pipeline/
│   ├── transcribe/            # Google STT (Chirp 2/3) and WhisperX backends
│   ├── format/                # Transcript → ASS subtitle formatting
│   ├── translate/             # LLM-based subtitle translation
│   ├── postprocess/           # Profile-driven editorial cleanup
│   └── report/                # HTML review report generator
profiles/                      # TOML profiles (examples/ tracked, local/ gitignored)
prompts/                       # Legacy prompt files (DEPRECATED — inline into profile TOML instead)
tests/                         # pytest test suite
```

## Pipeline

1. **Transcribe** - Extract audio, send to STT backend, produce `transcript.json`
2. **Format** - Chunk words into subtitle lines, apply timing/extensions, produce `original.ass`
3. **Translate** - LLM translation of subtitles, produce `translated.ass`
4. **Postprocess** - Profile-driven cleanup of translated output

`autosub run` executes all four stages. Each stage can also run independently.

## Key Concepts

- **Profiles**: TOML files with `extends` inheritance. Searched: `profiles/local/` > `profiles/examples/` > `profiles/`. A profile not found anywhere is a hard error (`FileNotFoundError`), not a silent warning.
- **Prompts**: Inline TOML strings inside a profile's `[translate] prompt = ...`. A profile can use either:
  - **A single string** (`prompt = """..."""`) — one fragment.
  - **A list of strings** (`prompt = ["""...""", """..."""]`) — multiple fragments concatenated, in order. Useful for layering: e.g. one block for program identity, another for shared style rules, another for unit-specific context. Fragments inherit additively through `extends` — a child profile's fragments are appended to the parent's, never replacing them.
  - **(Deprecated)** A `.md`/`.txt` file path (`prompt = "prompts/foo.md"`). Still works but emits a deprecation warning. The `prompts/` directory will be removed in a future release.
- **Extensions**: `radio_discourse` (listener mail classification) and `corners` (program segment detection) run at format/postprocess time
- **Speaker maps**: Per-project TOML files mapping diarization labels to character names/colors
- **Config**: `config.toml` (gitignored) provides default CLI flags per stage

## Chirp 3 Backend (this branch)

- `--backend chirp_3` selects Chirp 3 transcription
- Requires Opus encoding (WAV/AAC return empty results — observed behavior, not documented)
- Audio > 18 min is split into chunks and transcribed in parallel via `ThreadPoolExecutor`
- Chirp 3 returns bogus word timestamps at internal 18-min boundaries; `_clamp_word_timestamps` fixes these before offset is applied
- SpeechAdaptation PhraseSet is incompatible with `enable_word_time_offsets` on Chirp 3, so vocabulary hints are skipped (logged as warning)
- Chirp 2 remains the default and uses WAV (pcm_s16le) encoding

## Conventions

- Commit messages: imperative mood, lowercase, concise
- Branch per feature - don't tangle unrelated features
- Local/user-specific files go in gitignored dirs (`profiles/local/`, `config.toml`). `prompts/local/` exists for legacy file-based prompts only — new content should be inlined directly into the profile TOML.
