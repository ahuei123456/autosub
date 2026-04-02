import typer
import logging
from pathlib import Path
from typing import Sequence

from click.core import ParameterSource

from autosub.core.cli_config import (
    CLIConfigError,
    apply_command_config,
    apply_run_config,
    load_cli_config,
)
from autosub.core.llm import ReasoningEffort
from autosub.core.utils import parse_timestamp
from autosub.pipeline.transcribe import main as transcribe_main
from autosub.pipeline.format import main as format_module
from autosub.pipeline.postprocess import main as postprocess_module
from autosub.pipeline.translate import main as translate_module
from autosub.core.profile import load_unified_profile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="AutoSub CLI for Japanese subtitle generation and translation")


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


def _infer_llm_provider(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("gemini"):
        return "google-vertex"
    if normalized.startswith(("gpt", "o", "chatgpt")):
        return "openai"
    raise typer.BadParameter(
        f"Could not infer an LLM provider from model '{model_name}'. "
        "Use --llm-provider explicitly."
    )


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
    Extracts audio and transcribes Japanese speech with Google Cloud Speech-to-Text.
    """
    resolved = apply_command_config(
        ctx,
        "transcribe",
        {
            "output": output,
            "language": language,
            "vocab": vocab,
            "profile": profile,
            "start": start,
            "end": end,
        },
    )
    output = resolved["output"]
    language = resolved["language"]
    vocab = resolved["vocab"]
    profile = resolved["profile"]
    start = resolved["start"]
    end = resolved["end"]
    time_ranges = _normalize_time_ranges(start, end)

    logger.info(f"Starting transcription pipeline for: {video_path}")

    final_vocab = []
    if profile:
        profile_data = load_unified_profile(profile)
        final_vocab.extend(profile_data["vocab"])
    if vocab:
        final_vocab.extend(vocab)

    try:
        result = transcribe_main.transcribe(
            video_path,
            output,
            language,
            final_vocab,
            time_ranges=time_ranges,
        )
        logger.info(f"Success! Saved {len(result.words)} words to {output}")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
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
    replacements = {}
    if profile:
        profile_data = load_unified_profile(profile)
        timing_config = profile_data.get("timing", {})
        extensions_config = profile_data.get("extensions", {})
        replacements = profile_data.get("replacements", {})

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
            replacements=replacements,
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
        help="LLM provider to use for the vertex engine ('google-vertex', 'anthropic', or 'openai').",
    ),
    vertex_reasoning_effort: ReasoningEffort | None = typer.Option(
        "medium",
        "--vertex-reasoning-effort",
        help="Provider-agnostic reasoning effort for LLM translation ('off', 'minimal', 'low', 'medium', or 'high').",
    ),
    vertex_reasoning_budget: int | None = typer.Option(
        None,
        "--vertex-reasoning-budget",
        help="Optional token-budget override for Vertex LLM reasoning. For Gemini 2.5 this maps directly to thinking budget; for level-only models it is converted heuristically.",
    ),
    vertex_reasoning_dynamic: bool | None = typer.Option(
        None,
        "--vertex-reasoning-dynamic/--no-vertex-reasoning-dynamic",
        help="Request dynamic reasoning budget when the selected Vertex model family supports it.",
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

    final_prompt_parts = []
    if profile:
        profile_data = load_unified_profile(profile)
        final_prompt_parts.extend(profile_data["prompt"])

        if profile_data.get("glossary"):
            glossary_text = "Glossary (Always translate these exact phrases):\n"
            for ja, en in profile_data["glossary"].items():
                glossary_text += f'- "{ja}" -> "{en}"\n'
            final_prompt_parts.append(glossary_text)

    if prompt:
        final_prompt_parts.append(prompt)

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None

    resolved_engine = engine
    resolved_provider = llm_provider
    if vertex_model:
        if engine == "cloud-v3":
            raise typer.BadParameter(
                "--model cannot be used with --engine cloud-v3.",
                param_hint="--model",
            )
        resolved_engine = "vertex"
        resolved_provider = _infer_llm_provider(vertex_model)

    try:
        translate_module.translate_subtitles(
            input_ass,
            out,
            engine=resolved_engine,
            system_prompt=final_prompt,
            target_lang=target_lang,
            source_lang=source_lang,
            bilingual=bilingual,
            model=vertex_model,
            location=vertex_location,
            provider=resolved_provider,
            reasoning_effort=vertex_reasoning_effort,
            reasoning_budget_tokens=vertex_reasoning_budget,
            reasoning_dynamic=vertex_reasoning_dynamic,
            chunk_size=chunk_size,
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
        final_extensions = profile_data.get("extensions", {})

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
        "--vertex-reasoning-effort",
        help="Provider-agnostic reasoning effort for LLM translation ('off', 'minimal', 'low', 'medium', or 'high').",
    ),
    llm_provider: str = typer.Option(
        "google-vertex",
        "--llm-provider",
        help="LLM provider to use for translation ('google-vertex', 'anthropic', or 'openai').",
    ),
    bilingual: bool = typer.Option(
        False, "--bilingual/--replace", help="Include original text on top."
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

    transcript_out = out_dir / "transcript.json"
    original_ass_out = out_dir / "original.ass"
    translated_ass_out = out_dir / "translated.ass"

    # Resolve Profile
    final_vocab = []
    final_prompt_parts = []
    final_timing = {}
    final_extensions = {}
    replacements = {}
    if profile:
        profile_data = load_unified_profile(profile)
        final_vocab.extend(profile_data["vocab"])
        final_prompt_parts.extend(profile_data["prompt"])
        final_timing = profile_data.get("timing", {})
        final_extensions = profile_data.get("extensions", {})
        replacements = profile_data.get("replacements", {})

        if profile_data.get("glossary"):
            glossary_text = "Glossary (Always translate these exact phrases):\n"
            for ja, en in profile_data["glossary"].items():
                glossary_text += f'- "{ja}" -> "{en}"\n'
            final_prompt_parts.append(glossary_text)

    if vocab:
        final_vocab.extend(vocab)
    if prompt:
        final_prompt_parts.append(prompt)

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None
    resolved_provider = _infer_llm_provider(model) if model else llm_provider

    # Step 1: Transcribe
    try:
        logger.info("[Step 1/4] Transcribing...")
        transcribe_main.transcribe(
            video_path,
            transcript_out,
            language,
            final_vocab,
            time_ranges=time_ranges,
        )
    except Exception as e:
        logger.error(f"Failed during transcription: {e}")
        raise typer.Exit(code=1)

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
            extensions_config=final_extensions,
            replacements=replacements,
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
            model=model,
            provider=resolved_provider,
            reasoning_effort=vertex_reasoning_effort,
            chunk_size=chunk_size,
        )
    except Exception as e:
        logger.error(f"Failed during translation: {e}")
        raise typer.Exit(code=1)

    # Step 4: Postprocess
    try:
        logger.info("[Step 4/4] Postprocessing...")
        postprocess_module.postprocess_subtitles(
            translated_ass_out,
            extensions_config=final_extensions,
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
