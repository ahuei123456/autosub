import copy
import typer
import logging
import tomllib
from pathlib import Path
from typing import Sequence

from click.core import ParameterSource

from autosub.core.cli_config import (
    CLIConfigError,
    apply_command_config,
    apply_run_config,
    load_cli_config,
)
from autosub.core.llm import LLMResolutionError, ReasoningEffort, resolve_llm_selection
from autosub.core.utils import parse_timestamp
from autosub.pipeline.transcribe import main as transcribe_main
from autosub.pipeline.format import main as format_module
from autosub.pipeline.postprocess import main as postprocess_module
from autosub.pipeline.translate import main as translate_module
from autosub.core.profile import load_unified_profile

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

PROFILE_LOAD_ERRORS = (ValueError, FileNotFoundError, tomllib.TOMLDecodeError)

app = typer.Typer(help="AutoSub CLI for Japanese subtitle generation and translation")


def _add_file_logger(log_path: Path) -> None:
    """Add a FileHandler to the root logger so all output is also saved to disk."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # allow DEBUG through to file handler
    # Keep console at INFO
    for h in root.handlers:
        h.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    logger.info(f"Saving log to {log_path}")


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Path = typer.Option(
        Path("config.toml"),
        "--config",
        dir_okay=False,
        help="Load CLI defaults from this TOML file. Defaults to ./config.toml when present.",
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Ignore config.toml defaults for this invocation.",
    ),
):
    ctx.ensure_object(dict)

    if no_config:
        ctx.obj["cli_config"] = {}
        return

    if config_path.exists():
        try:
            ctx.obj["cli_config"] = load_cli_config(config_path)
        except CLIConfigError as exc:
            raise typer.BadParameter(str(exc), param_hint="--config") from exc
        return

    if ctx.get_parameter_source("config_path") != ParameterSource.DEFAULT:
        raise typer.BadParameter(
            f"Config file not found: {config_path}", param_hint="--config"
        )

    ctx.obj["cli_config"] = {}


def _command_option_has_config_override(
    ctx: typer.Context, command_name: str, parameter_name: str
) -> bool:
    cli_config = (ctx.obj or {}).get("cli_config", {})
    return parameter_name in cli_config.get(command_name, {})


def _run_option_has_config_override(ctx: typer.Context, parameter_name: str) -> bool:
    cli_config = (ctx.obj or {}).get("cli_config", {})
    return parameter_name in cli_config.get(
        "run", {}
    ) or parameter_name in cli_config.get("translate", {})


def _resolve_model_selection_or_exit(
    *,
    model: str,
    provider: str | None,
) -> tuple[str, str]:
    try:
        selection = resolve_llm_selection(model=model, provider=provider)
    except LLMResolutionError as exc:
        raise typer.BadParameter(str(exc), param_hint="--model") from exc
    return selection.provider, selection.model or model


def _exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def _extract_format_profile_config(profile_data: dict) -> tuple[dict, dict, dict]:
    format_profile = profile_data.get("format", {})
    timing_config = {
        key: value
        for key, value in format_profile.items()
        if key not in {"extensions", "replacements", "normalizer"}
    }
    replacements = format_profile.get("replacements", {})
    normalizer_config = format_profile.get("normalizer", {})
    if normalizer_config:
        normalizer_config = copy.deepcopy(normalizer_config)
        engine = str(normalizer_config.get("engine", "exact")).lower()
        normalizer_config["engine"] = engine
        if engine == "exact":
            normalizer_config.setdefault("replacements", copy.deepcopy(replacements))
        elif engine == "llm":
            if replacements:
                raise ValueError(
                    '[format.replacements] cannot be combined with [format.normalizer] engine="llm".'
                )
        else:
            raise ValueError(
                f"Unsupported format.normalizer.engine '{engine}'. Expected 'exact' or 'llm'."
            )
    elif replacements:
        normalizer_config = {
            "engine": "exact",
            "replacements": copy.deepcopy(replacements),
        }
    return (
        timing_config,
        format_profile.get("extensions", {}),
        normalizer_config,
    )


def _build_glossary_prompt(glossary: dict[str, str]) -> str | None:
    if not glossary:
        return None

    glossary_text = "Glossary (Always translate these exact phrases):\n"
    for source_text, translated_text in glossary.items():
        glossary_text += f'- "{source_text}" -> "{translated_text}"\n'
    return glossary_text


def _coerce_time_values(
    value: str | Sequence[str] | None,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _validate_time_range(
    start_time: str | None, end_time: str | None, range_number: int | None = None
) -> None:
    label_prefix = f"Range {range_number}: " if range_number is not None else ""
    try:
        start_seconds = parse_timestamp(start_time) if start_time is not None else None
        end_seconds = parse_timestamp(end_time) if end_time is not None else None
    except ValueError as exc:
        raise typer.BadParameter(
            f"{label_prefix}{exc}", param_hint="--start/--end"
        ) from exc

    if (
        start_seconds is not None
        and end_seconds is not None
        and end_seconds <= start_seconds
    ):
        raise typer.BadParameter(
            f"{label_prefix}end time must be greater than start time.",
            param_hint="--start/--end",
        )


def _normalize_time_ranges(
    start: str | Sequence[str] | None, end: str | Sequence[str] | None
) -> list[tuple[str | None, str | None]]:
    starts = _coerce_time_values(start)
    ends = _coerce_time_values(end)

    if len(starts) <= 1 and len(ends) <= 1:
        start_time = starts[0] if starts else None
        end_time = ends[0] if ends else None
        _validate_time_range(start_time, end_time)
        return [(start_time, end_time)]

    if len(starts) != len(ends):
        raise typer.BadParameter(
            "When using repeated --start/--end flags, the number of starts and ends must match.",
            param_hint="--start/--end",
        )

    time_ranges: list[tuple[str | None, str | None]] = []
    for index, (start_time, end_time) in enumerate(zip(starts, ends), start=1):
        _validate_time_range(start_time, end_time, range_number=index)
        time_ranges.append((start_time, end_time))
    return time_ranges


def _transcript_hash(transcript_path: Path) -> str:
    """Compute a short hash of a transcript file for staleness detection."""
    import hashlib
    with open(transcript_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _interactive_speaker_review(
    result, speaker_map: dict[str, dict], output_path: Path,
    sample_count: int = 3,
    transcript_path: Path | None = None,
) -> dict[str, dict] | None:
    """Run interactive speaker label review and write assignments.

    Returns the loaded assignments dict, or None if no assignments were made.
    """
    from autosub.core.speaker_map import load_speaker_map

    # Group words by speaker label
    speaker_words: dict[str, list] = {}
    for w in result.words:
        label = w.speaker or "unknown"
        if label not in speaker_words:
            speaker_words[label] = []
        speaker_words[label].append(w)

    # Deduplicate real speakers from the map
    real_speakers: list[dict] = []
    seen_names: set[str] = set()
    for entry in speaker_map.values():
        if entry["name"] not in seen_names:
            real_speakers.append(entry)
            seen_names.add(entry["name"])

    if not real_speakers:
        logger.error("Speaker map has no speaker entries.")
        return None

    # Display samples per label
    sorted_labels = sorted(speaker_words.keys(), key=lambda x: (not x.isdigit(), x))
    typer.echo("\n=== Speaker Labels in Transcript ===\n")

    for label in sorted_labels:
        words = speaker_words[label]
        if len(words) <= sample_count * 10:
            indices = [0]
        else:
            step = len(words) // sample_count
            indices = [i * step for i in range(sample_count)]

        typer.echo(f"Speaker {label} ({len(words)} words):")
        for idx in indices:
            snippet_words = words[idx:idx + 10]
            t = snippet_words[0].start_time
            mins = int(t // 60)
            secs = t % 60
            text = "".join(w.word for w in snippet_words)
            if len(text) > 50:
                text = text[:47] + "..."
            typer.echo(f"  [{mins:02d}:{secs:04.1f}] {text}")
        typer.echo()

    # Interactive mapping
    typer.echo("=== Known Speakers ===\n")
    for i, sp in enumerate(real_speakers, 1):
        char = f" ({sp['character']})" if sp.get("character") else ""
        typer.echo(f"  {i}. {sp['name']}{char}")
    typer.echo()

    assignments: dict[str, dict] = {}
    for label in sorted_labels:
        while True:
            choice = typer.prompt(
                f"Assign label \"{label}\" → [1-{len(real_speakers)}/skip]",
                default="skip",
            )
            if choice.lower() == "skip":
                break
            try:
                idx = int(choice)
                if 1 <= idx <= len(real_speakers):
                    assignments[label] = real_speakers[idx - 1]
                    typer.echo(f"  → {real_speakers[idx - 1]['name']}")
                    break
                else:
                    typer.echo(f"  Enter 1-{len(real_speakers)} or 'skip'")
            except ValueError:
                typer.echo(f"  Enter 1-{len(real_speakers)} or 'skip'")

    if not assignments:
        logger.info("No assignments made.")
        return None

    # Log summary
    skipped = [l for l in sorted_labels if l not in assignments]
    logger.info(
        f"Assigned {len(assignments)}/{len(sorted_labels)} labels, "
        f"skipped {len(skipped)}: {skipped if skipped else 'none'}"
    )
    for label, entry in sorted(assignments.items()):
        word_count = len(speaker_words[label])
        logger.info(f"  Label {label} ({word_count} words) → {entry['name']}")

    # Write assignments file
    lines = [
        f"# Speaker assignments generated by review-speakers",
    ]
    if transcript_path:
        lines.append(f'transcript_hash = "{_transcript_hash(transcript_path)}"')
    lines.append("")
    for label, entry in sorted(assignments.items(), key=lambda x: x[0]):
        lines.append(f'[speakers."{label}"]')
        lines.append(f'name = "{entry["name"]}"')
        if entry.get("character"):
            lines.append(f'character = "{entry["character"]}"')
        if entry.get("color"):
            lines.append(f'color = "{entry["color"]}"')
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote speaker assignments to {output_path}")
    return load_speaker_map(output_path)


@app.command()
def transcribe(
    ctx: typer.Context,
    video_path: Path = typer.Argument(
        ..., help="Path to the input video or audio file"
    ),
    output: Path = typer.Option(
        Path("transcript.json"),
        "--out",
        "-o",
        help="Path to save the output JSON transcript",
    ),
    language: str = typer.Option(
        "ja-JP",
        "--language",
        "-l",
        help="Language code for transcription (e.g. ja-JP, en-US)",
    ),
    vocab: list[str] = typer.Option(
        None,
        "--vocab",
        "-v",
        help="Custom context terminology to increase probability of recognition (can be passed multiple times).",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load transcription vocabulary hints.",
    ),
    transcription_backend: str = typer.Option(
        "chirp_2",
        "--backend",
        "--transcription-backend",
        help="Transcription backend to use ('chirp_2', 'chirp_3', or 'whisperx').",
    ),
    whisper_model: str = typer.Option(
        "large-v2",
        "--whisper-model",
        help="WhisperX model name when --backend whisperx is selected.",
    ),
    whisper_device: str = typer.Option(
        "cpu",
        "--whisper-device",
        help="Device for WhisperX transcription ('cpu' or 'cuda').",
    ),
    whisper_compute_type: str = typer.Option(
        "int8",
        "--whisper-compute-type",
        help="CTranslate2 compute type for WhisperX (for example 'int8' or 'float16').",
    ),
    whisper_batch_size: int = typer.Option(
        16,
        "--whisper-batch-size",
        min=1,
        help="Batch size for WhisperX transcription.",
    ),
    whisper_diarize: bool = typer.Option(
        False,
        "--whisper-diarize/--no-whisper-diarize",
        help="Enable WhisperX diarization and populate speaker labels.",
    ),
    whisper_hf_token: str = typer.Option(
        None,
        "--whisper-hf-token",
        help="Optional Hugging Face token for WhisperX diarization.",
    ),
    speakers: int = typer.Option(
        None,
        "--speakers",
        "-s",
        help="Number of speakers for diarization. Enables per-word speaker labels in the transcript.",
    ),
    start: list[str] = typer.Option(
        None,
        "--start",
        help="Start time for transcription (e.g. 00:01:00 or 60). Can be passed multiple times and pairs by order with --end.",
    ),
    end: list[str] = typer.Option(
        None,
        "--end",
        help="End time for transcription (e.g. 00:04:00 or 240). Can be passed multiple times and pairs by order with --start.",
    ),
):
    """
    Extracts audio and transcribes speech with the selected backend.
    """
    resolved = apply_command_config(
        ctx,
        "transcribe",
        {
            "output": output,
            "language": language,
            "vocab": vocab,
            "profile": profile,
            "transcription_backend": transcription_backend,
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "whisper_compute_type": whisper_compute_type,
            "whisper_batch_size": whisper_batch_size,
            "whisper_diarize": whisper_diarize,
            "whisper_hf_token": whisper_hf_token,
            "start": start,
            "end": end,
        },
    )
    output = resolved["output"]
    language = resolved["language"]
    vocab = resolved["vocab"]
    profile = resolved["profile"]
    transcription_backend = resolved["transcription_backend"]
    whisper_model = resolved["whisper_model"]
    whisper_device = resolved["whisper_device"]
    whisper_compute_type = resolved["whisper_compute_type"]
    whisper_batch_size = resolved["whisper_batch_size"]
    whisper_diarize = resolved["whisper_diarize"]
    whisper_hf_token = resolved["whisper_hf_token"]
    start = resolved["start"]
    end = resolved["end"]
    time_ranges = _normalize_time_ranges(start, end)

    logger.info(f"Starting transcription pipeline for: {video_path}")

    final_vocab = []
    if profile:
        profile_data = load_unified_profile(profile)
        final_vocab.extend(profile_data.get("transcribe", {}).get("vocab", []))
    if vocab:
        final_vocab.extend(vocab)

    try:
        result = transcribe_main.transcribe(
            video_path,
            output,
            language,
            final_vocab,
            time_ranges=time_ranges,
            transcription_backend=transcription_backend,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_batch_size=whisper_batch_size,
            whisper_diarize=whisper_diarize,
            whisper_hf_token=whisper_hf_token,
            num_speakers=speakers,
        )
        logger.info(f"Success! Saved {len(result.words)} words to {output}")
    except Exception as e:
        logger.exception("Error during transcription (%s)", _exception_summary(e))
        raise typer.Exit(code=1)


@app.command()
def format(
    ctx: typer.Context,
    input_transcript: Path = typer.Argument(
        ...,
        help="Path to the input transcript.json file generated by the transcribe command.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path to save the generated output .ass file (defaults to original.ass in the same directory).",
    ),
    keyframes: Path = typer.Option(
        None, "--keyframes", help="Path to Aegisub keyframes log for snapping."
    ),
    fps: float = typer.Option(
        0.0, "--fps", help="Video FPS, required if --keyframes is used."
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load timing and formatting extension configuration.",
    ),
    speaker_map: Path = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker_map.toml mapping API speaker labels to character names and colors.",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Step 2: Converts a transcript JSON into timed single-lane .ass subtitles.
    """
    resolved = apply_command_config(
        ctx,
        "format",
        {
            "out": out,
            "keyframes": keyframes,
            "fps": fps,
            "profile": profile,
        },
    )
    out = resolved["out"]
    keyframes = resolved["keyframes"]
    fps = resolved["fps"]
    profile = resolved["profile"]

    if not out:
        out = input_transcript.with_name("original.ass")

    if keyframes and fps <= 0:
        logger.error("--fps is required when --keyframes is provided.")
        raise typer.Exit(code=1)

    timing_config = {}
    extensions_config = {}
    normalizer_config = {}
    if profile:
        try:
            profile_data = load_unified_profile(profile)
            timing_config, extensions_config, normalizer_config = (
                _extract_format_profile_config(profile_data)
            )
        except PROFILE_LOAD_ERRORS as e:
            logger.error(f"Error while loading format profile: {e}")
            raise typer.Exit(code=1)

    loaded_speaker_map = None
    if speaker_map:
        from autosub.core.speaker_map import load_speaker_map
        loaded_speaker_map = load_speaker_map(speaker_map)

    kf_ms = None
    if keyframes and fps > 0:
        from autosub.pipeline.video.keyframes import parse_aegisub_keyframes

        kf_ms = parse_aegisub_keyframes(keyframes, fps)

    try:
        format_module.format_subtitles(
            input_transcript,
            out,
            keyframes=kf_ms,
            timing_config=timing_config,
            extensions_config=extensions_config,
            normalizer_config=normalizer_config,
            speaker_map=loaded_speaker_map,
        )
    except Exception as e:
        logger.error(f"Error during formatting: {e}")
        raise typer.Exit(code=1)


@app.command()
def translate(
    ctx: typer.Context,
    input_ass: Path = typer.Argument(
        ...,
        help="Path to the original .ass file generated by the format command.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path to save the translated .ass file (defaults to translated.ass in the same directory).",
    ),
    engine: str = typer.Option(
        "vertex",
        "--engine",
        "-e",
        help="Translation engine to use ('vertex' or 'cloud-v3').",
    ),
    prompt: str = typer.Option(
        None, "--prompt", "-p", help="System prompt to guide the LLM translation."
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load translation prompt settings.",
    ),
    target_lang: str = typer.Option("en", "--target", help="Target language code."),
    source_lang: str = typer.Option("ja", "--source", help="Source language code."),
    vertex_model: str | None = typer.Option(
        None,
        "--model",
        "--llm-model",
        "--vertex-model",
        help="LLM model name for translation. Also infers the provider for Gemini, Claude, and OpenAI model names.",
    ),
    vertex_location: str = typer.Option(
        "global",
        "--llm-location",
        "--vertex-location",
        help="LLM location or region for translation.",
    ),
    llm_provider: str = typer.Option(
        "google-vertex",
        "--llm-provider",
        help="LLM provider to use for the vertex engine ('google-vertex', 'anthropic-vertex', 'anthropic', 'openai', or 'openrouter').",
    ),
    vertex_reasoning_effort: ReasoningEffort | None = typer.Option(
        "medium",
        "--llm-reasoning-effort",
        "--vertex-reasoning-effort",
        help="Provider-agnostic reasoning effort for LLM translation ('off', 'minimal', 'low', 'medium', or 'high').",
    ),
    vertex_reasoning_budget: int | None = typer.Option(
        None,
        "--llm-reasoning-budget",
        "--vertex-reasoning-budget",
        help="Optional token-budget override for LLM reasoning. For Gemini 2.5 this maps directly to thinking budget; for level-only models it is converted heuristically.",
    ),
    vertex_reasoning_dynamic: bool | None = typer.Option(
        None,
        "--llm-reasoning-dynamic/--no-llm-reasoning-dynamic",
        "--vertex-reasoning-dynamic/--no-vertex-reasoning-dynamic",
        help="Request dynamic reasoning budget when the selected provider and model family support it.",
    ),
    bilingual: bool = typer.Option(
        False,
        "--bilingual/--replace",
        help="Include original text on top, or replace completely.",
    ),
    chunk_size: int = typer.Option(
        0,
        "--chunk-size",
        min=0,
        help="Number of subtitle lines per chunk. Use 0 to disable chunking.",
    ),
    mark_chunks: bool = typer.Option(
        False,
        "--mark-chunks/--no-mark-chunks",
        help="Insert comment events at artificial chunk boundaries for review.",
    ),
    save_log: bool = typer.Option(
        False,
        "--save-log/--no-save-log",
        help="Save full log output to a .log file next to the output file.",
    ),
    retry_chunk: list[int] = typer.Option(
        None,
        "--retry-chunk",
        help="Re-translate specific chunk(s) by number (1-based). Can be passed multiple times.",
    ),
    speaker_map: Path = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker_map.toml to inject speaker identity context into the translation prompt.",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Step 3: Translates a .ass subtitle file using the configured Translation Engine.
    """
    resolved = apply_command_config(
        ctx,
        "translate",
        {
            "out": out,
            "engine": engine,
            "prompt": prompt,
            "profile": profile,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "vertex_model": vertex_model,
            "vertex_location": vertex_location,
            "llm_provider": llm_provider,
            "vertex_reasoning_effort": vertex_reasoning_effort,
            "vertex_reasoning_budget": vertex_reasoning_budget,
            "vertex_reasoning_dynamic": vertex_reasoning_dynamic,
            "bilingual": bilingual,
            "chunk_size": chunk_size,
        },
    )
    out = resolved["out"]
    engine = resolved["engine"]
    prompt = resolved["prompt"]
    profile = resolved["profile"]
    target_lang = resolved["target_lang"]
    source_lang = resolved["source_lang"]
    vertex_model = resolved["vertex_model"]
    vertex_location = resolved["vertex_location"]
    llm_provider = resolved["llm_provider"]
    vertex_reasoning_effort = resolved["vertex_reasoning_effort"]
    vertex_reasoning_budget = resolved["vertex_reasoning_budget"]
    vertex_reasoning_dynamic = resolved["vertex_reasoning_dynamic"]
    bilingual = resolved["bilingual"]
    chunk_size = resolved["chunk_size"]

    if not out:
        out = input_ass.with_name("translated.ass")

    translate_log_dir = None
    if save_log:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_stem = input_ass.stem.removesuffix("_original")
        translate_log_dir = input_ass.parent / f"{log_stem}_logs_{ts}"
        translate_log_dir.mkdir(parents=True, exist_ok=True)
        _add_file_logger(translate_log_dir / "run.log")

    loaded_speaker_map = None
    if speaker_map:
        from autosub.core.speaker_map import load_speaker_map
        loaded_speaker_map = load_speaker_map(speaker_map)

    final_prompt_parts = []
    if profile:
        profile_data = load_unified_profile(profile)
        translate_profile = profile_data.get("translate", {})
        final_prompt_parts.extend(translate_profile.get("prompt", []))
        glossary_text = _build_glossary_prompt(translate_profile.get("glossary", {}))
        if glossary_text:
            final_prompt_parts.append(glossary_text)

    if prompt:
        final_prompt_parts.append(prompt)

    if loaded_speaker_map:
        from autosub.core.speaker_map import build_speaker_prompt
        final_prompt_parts.append(build_speaker_prompt(loaded_speaker_map))

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None

    resolved_engine = engine
    resolved_provider = llm_provider
    resolved_model = vertex_model
    provider_override = None
    if ctx.get_parameter_source(
        "llm_provider"
    ) != ParameterSource.DEFAULT or _command_option_has_config_override(
        ctx, "translate", "llm_provider"
    ):
        provider_override = llm_provider
    if vertex_model:
        if engine == "cloud-v3":
            raise typer.BadParameter(
                "--model cannot be used with --engine cloud-v3.",
                param_hint="--model",
            )
        resolved_engine = "vertex"
        resolved_provider, resolved_model = _resolve_model_selection_or_exit(
            model=vertex_model,
            provider=provider_override,
        )

    try:
        translate_module.translate_subtitles(
            input_ass,
            out,
            engine=resolved_engine,
            system_prompt=final_prompt,
            target_lang=target_lang,
            source_lang=source_lang,
            bilingual=bilingual,
            model=resolved_model,
            location=vertex_location,
            provider=resolved_provider,
            reasoning_effort=vertex_reasoning_effort,
            reasoning_budget_tokens=vertex_reasoning_budget,
            reasoning_dynamic=vertex_reasoning_dynamic,
            chunk_size=chunk_size,
            debug=mark_chunks,
            retry_chunks=retry_chunk or None,
            log_dir=translate_log_dir,
        )
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        raise typer.Exit(code=1)


@app.command()
def postprocess(
    ctx: typer.Context,
    input_ass: Path = typer.Argument(
        ...,
        help="Path to the translated .ass file generated by the translate command.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path to save the postprocessed .ass file (defaults to overwriting the input file).",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load postprocessing extension configuration.",
    ),
    bilingual: bool = typer.Option(
        False,
        "--bilingual/--replace",
        help="Treat the input as bilingual stacked subtitles, or translated-only subtitles.",
    ),
):
    """
    Step 4: Applies optional post-translation subtitle cleanup and editorial transforms.
    """
    resolved = apply_command_config(
        ctx,
        "postprocess",
        {
            "out": out,
            "profile": profile,
            "bilingual": bilingual,
        },
    )
    out = resolved["out"]
    profile = resolved["profile"]
    bilingual = resolved["bilingual"]

    final_extensions = {}
    if profile:
        profile_data = load_unified_profile(profile)
        final_extensions = profile_data.get("postprocess", {}).get("extensions", {})

    try:
        postprocess_module.postprocess_subtitles(
            input_ass,
            output_ass_path=out,
            extensions_config=final_extensions,
            bilingual=bilingual,
        )
    except Exception as e:
        logger.error(f"Error during postprocessing: {e}")
        raise typer.Exit(code=1)


@app.command(name="review-speakers")
def review_speakers(
    transcript: Path = typer.Argument(
        ...,
        help="Path to the transcript JSON file from the transcribe step.",
        exists=True,
        dir_okay=False,
    ),
    speaker_map_path: Path = typer.Option(
        ...,
        "--speaker-map",
        help="Path to speaker_map.toml with the real speaker identities (not modified).",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        None,
        "--out",
        help="Output path for speaker assignments (default: speaker_assignments.toml next to transcript).",
    ),
    sample_lines: int = typer.Option(
        3,
        "--sample-lines",
        help="Number of sample lines to show per speaker label.",
    ),
):
    """
    Review diarized speaker labels and interactively map them to real speakers.

    Shows sample transcript lines per speaker label, then prompts you to assign
    each label to a speaker from the speaker map. Writes assignments to a
    separate file (speaker_assignments.toml) — the original speaker map is not modified.
    """
    import json

    from autosub.core.schemas import TranscriptionResult
    from autosub.core.speaker_map import load_speaker_map

    logger.info(f"Loading transcript from {transcript}...")
    with open(transcript, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = TranscriptionResult(**data)

    if not result.words:
        logger.error("Transcript has no words.")
        raise typer.Exit(code=1)

    logger.info(f"Loaded {len(result.words)} words from transcript.")

    existing_map = load_speaker_map(speaker_map_path)

    if not output:
        output = transcript.parent / "speaker_assignments.toml"

    assigned = _interactive_speaker_review(
        result, existing_map, output, sample_count=sample_lines,
        transcript_path=transcript,
    )
    if not assigned:
        raise typer.Exit()

    logger.info(f"Use with: autosub format ... --speaker-map {output}")


@app.command()
def run(
    ctx: typer.Context,
    video_path: Path = typer.Argument(
        ..., help="Path to the input video or audio file"
    ),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        help="Directory to save generated files. Defaults to the input video's directory.",
    ),
    language: str = typer.Option(
        "ja-JP", "--language", "-l", help="Language code for transcription (e.g. ja-JP)"
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Profile name to load vocabulary, timing, prompt, and extension settings.",
    ),
    vocab: list[str] = typer.Option(
        None, "--vocab", "-v", help="Custom transcription hints."
    ),
    transcription_backend: str = typer.Option(
        "chirp_2",
        "--backend",
        "--transcription-backend",
        help="Transcription backend to use ('chirp_2', 'chirp_3', or 'whisperx').",
    ),
    whisper_model: str = typer.Option(
        "large-v2",
        "--whisper-model",
        help="WhisperX model name when --backend whisperx is selected.",
    ),
    whisper_device: str = typer.Option(
        "cpu",
        "--whisper-device",
        help="Device for WhisperX transcription ('cpu' or 'cuda').",
    ),
    whisper_compute_type: str = typer.Option(
        "int8",
        "--whisper-compute-type",
        help="CTranslate2 compute type for WhisperX (for example 'int8' or 'float16').",
    ),
    whisper_batch_size: int = typer.Option(
        16,
        "--whisper-batch-size",
        min=1,
        help="Batch size for WhisperX transcription.",
    ),
    whisper_diarize: bool = typer.Option(
        False,
        "--whisper-diarize/--no-whisper-diarize",
        help="Enable WhisperX diarization and populate speaker labels.",
    ),
    whisper_hf_token: str = typer.Option(
        None,
        "--whisper-hf-token",
        help="Optional Hugging Face token for WhisperX diarization.",
    ),
    prompt: str = typer.Option(
        None, "--prompt", "-p", help="System prompt to guide the LLM translation."
    ),
    target_lang: str = typer.Option("en", "--target", help="Target language code."),
    source_lang: str = typer.Option("ja", "--source", help="Source language code."),
    model: str | None = typer.Option(
        None,
        "--model",
        help="LLM model name for translation. Also infers the provider for Gemini, Claude, and OpenAI model names.",
    ),
    vertex_reasoning_effort: ReasoningEffort | None = typer.Option(
        "medium",
        "--llm-reasoning-effort",
        "--vertex-reasoning-effort",
        help="Provider-agnostic reasoning effort for LLM translation ('off', 'minimal', 'low', 'medium', or 'high').",
    ),
    llm_provider: str = typer.Option(
        "google-vertex",
        "--llm-provider",
        help="LLM provider to use for translation ('google-vertex', 'anthropic-vertex', 'anthropic', 'openai', or 'openrouter').",
    ),
    bilingual: bool = typer.Option(
        False, "--bilingual/--replace", help="Include original text on top."
    ),
    speakers: int = typer.Option(
        None,
        "--speakers",
        "-s",
        help="Number of speakers for diarization. Enables per-word speaker labels and per-speaker styles.",
    ),
    keyframes: Path = typer.Option(
        None,
        "--keyframes",
        help="Path to existing keyframes log (overrides extraction).",
    ),
    extract_keyframes: bool = typer.Option(
        True,
        "--extract-keyframes/--no-extract-keyframes",
        help="Automatically extract keyframes when optional external tooling is installed.",
    ),
    start: list[str] = typer.Option(
        None,
        "--start",
        help="Start time for transcription (e.g. 00:01:00 or 60). Can be passed multiple times and pairs by order with --end.",
    ),
    end: list[str] = typer.Option(
        None,
        "--end",
        help="End time for transcription (e.g. 00:04:00 or 240). Can be passed multiple times and pairs by order with --start.",
    ),
    chunk_size: int = typer.Option(
        0,
        "--chunk-size",
        min=0,
        help="Number of subtitle lines per chunk. Use 0 to disable chunking.",
    ),
    mark_chunks: bool = typer.Option(
        False,
        "--mark-chunks/--no-mark-chunks",
        help="Insert comment events at artificial chunk boundaries for review.",
    ),
    save_log: bool = typer.Option(
        False,
        "--save-log/--no-save-log",
        help="Save full log output to a .log file in the output directory.",
    ),
    retry_chunk: list[int] = typer.Option(
        None,
        "--retry-chunk",
        help="Re-translate specific chunk(s) by number (1-based). Can be passed multiple times.",
    ),
    speaker_map: Path = typer.Option(
        None,
        "--speaker-map",
        help="Path to speaker_map.toml mapping API speaker labels to character names and colors.",
        exists=True,
        dir_okay=False,
    ),
):
    """
    Runs the end-to-end Japanese pipeline (Transcribe -> Format -> Translate -> Postprocess).
    """
    resolved = apply_run_config(
        ctx,
        {
            "out_dir": out_dir,
            "language": language,
            "profile": profile,
            "vocab": vocab,
            "transcription_backend": transcription_backend,
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "whisper_compute_type": whisper_compute_type,
            "whisper_batch_size": whisper_batch_size,
            "whisper_diarize": whisper_diarize,
            "whisper_hf_token": whisper_hf_token,
            "prompt": prompt,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "model": model,
            "vertex_reasoning_effort": vertex_reasoning_effort,
            "llm_provider": llm_provider,
            "bilingual": bilingual,
            "keyframes": keyframes,
            "extract_keyframes": extract_keyframes,
            "start": start,
            "end": end,
            "chunk_size": chunk_size,
        },
    )
    out_dir = resolved["out_dir"]
    language = resolved["language"]
    profile = resolved["profile"]
    vocab = resolved["vocab"]
    transcription_backend = resolved["transcription_backend"]
    whisper_model = resolved["whisper_model"]
    whisper_device = resolved["whisper_device"]
    whisper_compute_type = resolved["whisper_compute_type"]
    whisper_batch_size = resolved["whisper_batch_size"]
    whisper_diarize = resolved["whisper_diarize"]
    whisper_hf_token = resolved["whisper_hf_token"]
    prompt = resolved["prompt"]
    target_lang = resolved["target_lang"]
    source_lang = resolved["source_lang"]
    model = resolved["model"]
    vertex_reasoning_effort = resolved["vertex_reasoning_effort"]
    llm_provider = resolved["llm_provider"]
    bilingual = resolved["bilingual"]
    keyframes = resolved["keyframes"]
    extract_keyframes = resolved["extract_keyframes"]
    start = resolved["start"]
    end = resolved["end"]
    chunk_size = resolved["chunk_size"]
    time_ranges = _normalize_time_ranges(start, end)

    logger.info(f"Starting full autosub pipeline for: {video_path}")

    if not out_dir:
        out_dir = video_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    translate_log_dir = None
    if save_log:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        translate_log_dir = out_dir / f"{stem}_logs_{ts}"
        translate_log_dir.mkdir(parents=True, exist_ok=True)
        _add_file_logger(translate_log_dir / "run.log")

    transcript_out = out_dir / f"{stem}_transcript.json"
    original_ass_out = out_dir / f"{stem}_original.ass"
    translated_ass_out = out_dir / f"{stem}_translated.ass"

    # Resolve Profile
    final_vocab = []
    final_prompt_parts = []
    final_timing = {}
    final_format_extensions = {}
    final_postprocess_extensions = {}
    normalizer_config = {}
    if profile:
        try:
            profile_data = load_unified_profile(profile)
            transcribe_profile = profile_data.get("transcribe", {})
            translate_profile = profile_data.get("translate", {})
            postprocess_profile = profile_data.get("postprocess", {})
            final_vocab.extend(transcribe_profile.get("vocab", []))
            final_prompt_parts.extend(translate_profile.get("prompt", []))
            final_timing, final_format_extensions, normalizer_config = (
                _extract_format_profile_config(profile_data)
            )
            final_postprocess_extensions = postprocess_profile.get("extensions", {})
            if not speakers and profile_data.get("speakers"):
                speakers = profile_data["speakers"]
            glossary_text = _build_glossary_prompt(
                translate_profile.get("glossary", {})
            )
            if glossary_text:
                final_prompt_parts.append(glossary_text)
        except PROFILE_LOAD_ERRORS as e:
            logger.error(f"Failed while loading profile settings: {e}")
            raise typer.Exit(code=1)

    if vocab:
        final_vocab.extend(vocab)
    if prompt:
        final_prompt_parts.append(prompt)

    # Load speaker map early so it can be used in both prompt and format steps
    loaded_speaker_map = None
    if speaker_map:
        from autosub.core.speaker_map import load_speaker_map, build_speaker_prompt
        loaded_speaker_map = load_speaker_map(speaker_map)
        final_prompt_parts.append(build_speaker_prompt(loaded_speaker_map))

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None
    resolved_provider = llm_provider
    resolved_model = model
    provider_override = None
    if ctx.get_parameter_source(
        "llm_provider"
    ) != ParameterSource.DEFAULT or _run_option_has_config_override(
        ctx, "llm_provider"
    ):
        provider_override = llm_provider
    if model:
        resolved_provider, resolved_model = _resolve_model_selection_or_exit(
            model=model,
            provider=provider_override,
        )

    # Step 1: Transcribe
    try:
        logger.info("[Step 1/4] Transcribing...")
        result = transcribe_main.transcribe(
            video_path,
            transcript_out,
            language,
            final_vocab,
            time_ranges=time_ranges,
            transcription_backend=transcription_backend,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_batch_size=whisper_batch_size,
            whisper_diarize=whisper_diarize,
            whisper_hf_token=whisper_hf_token,
            replacements=replacements or None,
            num_speakers=speakers,
            replacements=replacements or None,
        )
    except Exception as e:
        logger.exception("Failed during transcription (%s)", _exception_summary(e))
        raise typer.Exit(code=1)

    # Interactive speaker review when extra labels are detected
    if speakers and result and loaded_speaker_map:
        unique_speakers = {w.speaker for w in result.words if w.speaker}
        if len(unique_speakers) > speakers:
            logger.warning(
                f"Requested {speakers} speakers but transcription returned "
                f"{len(unique_speakers)} labels: {sorted(unique_speakers)}."
            )
            assignments_path = transcript_out.parent / "speaker_assignments.toml"

            # Check for existing assignments from a previous review
            run_review = True
            if assignments_path.exists():
                import tomllib as _tomllib
                with open(assignments_path, "rb") as _f:
                    _adata = _tomllib.load(_f)
                stored_hash = _adata.get("transcript_hash")
                current_hash = _transcript_hash(transcript_out)
                if stored_hash == current_hash:
                    from autosub.core.speaker_map import load_speaker_map as _load_map
                    loaded_speaker_map = _load_map(assignments_path)
                    logger.info(f"Using existing speaker assignments from {assignments_path}")
                    run_review = False
                else:
                    logger.warning(
                        f"Speaker assignments in {assignments_path} are stale "
                        f"(transcript changed). Re-running speaker review."
                    )

            if run_review:
                logger.info("Starting interactive speaker review...")
                assigned = _interactive_speaker_review(
                    result, loaded_speaker_map, assignments_path,
                    transcript_path=transcript_out,
                )
                if assigned:
                    loaded_speaker_map = assigned

    # Step 1.5: Keyframes
    kf_ms = None
    vid_duration_ms = None
    try:
        logger.info("[Step 1.5/4] Processing Video Keyframes & Metadata...")
        from autosub.pipeline.video.keyframes import (
            get_fps,
            extract_keyframes as extract_keyframes_func,
            parse_aegisub_keyframes,
        )

        fps = get_fps(video_path)
        if fps > 0:
            import ffmpeg

            probe = ffmpeg.probe(str(video_path))
            try:
                vid_duration_ms = int(float(probe["format"]["duration"]) * 1000)
            except Exception:
                pass

            if keyframes and keyframes.exists():
                logger.info(f"Using provided keyframes: {keyframes}")
                kf_ms = parse_aegisub_keyframes(keyframes, fps)
            elif extract_keyframes:
                kf_out = out_dir / f"{video_path.stem}_keyframes.log"
                logger.info(f"Extracting keyframes to {kf_out}...")
                extract_keyframes_func(video_path, kf_out)
                kf_ms = parse_aegisub_keyframes(kf_out, fps)
        else:
            logger.warning("No video stream found. Keyframe extraction disabled.")
    except Exception as e:
        logger.warning(f"Failed to process keyframes: {e}")

    # Step 2: Format
    try:
        logger.info("[Step 2/4] Formatting...")
        format_module.format_subtitles(
            transcript_out,
            original_ass_out,
            keyframes=kf_ms,
            video_duration_ms=vid_duration_ms,
            timing_config=final_timing,
            extensions_config=final_format_extensions,
            normalizer_config=normalizer_config,
            speaker_map=loaded_speaker_map,
        )
    except Exception as e:
        logger.error(f"Failed during formatting: {e}")
        raise typer.Exit(code=1)

    # Step 3: Translate
    try:
        logger.info("[Step 3/4] Translating...")
        translate_module.translate_subtitles(
            original_ass_out,
            translated_ass_out,
            system_prompt=final_prompt,
            target_lang=target_lang,
            source_lang=source_lang,
            bilingual=bilingual,
            model=resolved_model,
            provider=resolved_provider,
            reasoning_effort=vertex_reasoning_effort,
            chunk_size=chunk_size,
            debug=mark_chunks,
            retry_chunks=retry_chunk or None,
            log_dir=translate_log_dir,
        )
    except Exception as e:
        logger.error(f"Failed during translation: {e}")
        raise typer.Exit(code=1)

    # Step 4: Postprocess
    try:
        logger.info("[Step 4/4] Postprocessing...")
        postprocess_module.postprocess_subtitles(
            translated_ass_out,
            extensions_config=final_postprocess_extensions,
            bilingual=bilingual,
        )
    except Exception as e:
        logger.error(f"Failed during postprocessing: {e}")
        raise typer.Exit(code=1)

    logger.info(
        f"Pipeline completed successfully! Final output saved to {translated_ass_out}"
    )


if __name__ == "__main__":
    app()
