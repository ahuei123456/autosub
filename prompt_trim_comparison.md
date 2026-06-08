# Prompt Trim — Before / After

Reducing per-chunk system-prompt size by removing rules that duplicate the hardcoded translator instructions in `autosub/pipeline/translate/translator.py` (`VertexTranslator._get_system_instruction`).

## Summary

| Layer | Before | After | Δ chars | Δ % |
|---|---|---|---|---|
| `prompts/examples/solo_seiyuu_radio.md` | ~2,800 | ~2,500 | −300 | −11% |
| `profiles/local/proseka/aftertalk.toml` (prompt strings only) | ~2,500 | ~1,920 | −580 | −23% |
| **Total per chunk** | **~5,300** | **~4,420** | **−880** | **−17%** |

Tokens saved per chunk: ~220. For a 30-min episode at 80-line chunks (~12 chunks), ~2,600 tokens saved per run.

What stayed put: the hardcoded translator rules in `translator.py` (kept for now — only file in this stack that lands upstream) and the ProSeka-specific rules in `aftertalk.toml` (all 8 rules preserved, just tighter wording).

---

## 1. `prompts/examples/solo_seiyuu_radio.md`

### Before (30 lines, ~2,800 chars)

```
You are translating a solo Japanese radio show hosted by a voice actress into natural English subtitles.

Core translation guidelines:
1. Treat the audio as spoken performance, not prose. The host is talking casually to listeners, reading messages, riffing on topics, and reacting in real time.
2. Prioritize natural, conversational English that sounds like a lively radio host. Do not produce stiff, overly literal, or essay-like phrasing.
3. Keep lines compact and subtitle-friendly. Prefer short clauses, contractions, and clean spoken English over exhaustive phrasing.
4. Preserve the host's persona. If she sounds playful, teasing, shy, awkward, excited, flustered, or sentimental, the English should carry that same energy.
5. Translate for listener experience, not dictionary equivalence. If a phrase has a routine radio-show meaning, use the natural English equivalent a fan subtitle reader would expect.
6. When she reads fan mail, questions, or topic submissions, make the listener's message sound like natural written-in spoken English, not stiff quoted prose.
7. When a line is host framing around a listener message, translate it as natural radio-host setup or wrap-up. Phrases like といただきました, とのことです, or similar quotative framing should usually become something like "Thanks for writing in," "That's what they sent in," or "That's what they said," depending on tone, rather than a literal quotative translation.
8. If a listener message and the host's framing or reaction are separated into different subtitle lines, keep them functionally separate in English too. Do not merge a listener sentence into a host framing line unless the split becomes unnatural in English.
9. Do not spell out filler sounds literally unless they add timing, hesitation, embarrassment, or comedic effect. Light hesitation can be rendered as "uh," "um," "well," or omitted entirely.
10. Preserve recurring bits, catchphrases, segment names, and self-introductions consistently. If a literal translation sounds awkward, prefer a natural recurring English rendering and keep it consistent across episodes.
11. Keep honorifics, titles, and franchise names only when they matter to fan understanding. Do not clutter the subtitle with unnecessary explanation.
12. If the host corrects herself, restarts a sentence, or speaks hesitantly, render it in natural spoken English, but do not add ellipses unless the transcript itself clearly shows an unfinished thought. When a thought simply continues into the next subtitle, prefer a comma or clean continuation instead.
13. When a joke or anecdote depends on phrasing, preserve the punchline or emotional beat even if sentence structure must change.
14. Assume the audience is familiar with anime, seiyuu, radio-show, and idol-fandom context. Translate clearly, but do not over-explain common fan culture references inside the subtitle itself.
15. Some radio shows follow a recurring structure with distinct corners/segments. When you recognize a cue phrase that signals a corner transition, adjust your translation tone and context accordingly.

Style constraints:
1. Use spoken English, not formal written English.
2. Avoid academic wording, passive constructions, and unnatural repetition.
3. Keep emotional exclamations lively but not exaggerated.
4. Maintain continuity across adjacent subtitle lines so one thought flows naturally into the next.
5. Do not use ellipses just to signal casual hesitation, filler, or subtitle line carryover.
```

### After (17 lines, ~2,500 chars)

```
You are translating a solo Japanese radio show hosted by a voice actress into English subtitles.

The audio is a casual spoken performance — the host talks to listeners, reads messages, riffs on topics, and reacts in real time. Translate for that context.

Radio-specific guidance:

1. Preserve the host's emotional register. If she sounds playful, teasing, shy, awkward, excited, flustered, or sentimental, the English should carry that same energy.
2. When she reads fan mail, questions, or topic submissions, make the listener's message sound like natural written-in spoken English — not stiff quoted prose.
3. When a line is host framing around a listener message, translate it as natural radio-host setup or wrap-up. Quotative phrases like といただきました, とのことです, or similar should become things like "Thanks for writing in," "That's what they sent in," or "That's what they said," depending on tone — not literal quotative translations.
4. If a listener message and the host's framing or reaction are on different subtitle lines, keep them functionally separate in English too. Do not merge a listener sentence into a host framing line unless the split becomes unnatural.
5. Do not spell out filler sounds literally unless they add timing, hesitation, embarrassment, or comedic effect. Light hesitation can be "uh," "um," "well," or omitted entirely.
6. Preserve recurring bits, catchphrases, segment names, and self-introductions consistently across episodes. If a literal translation sounds awkward, pick a natural recurring English rendering and keep it stable.
7. Keep honorifics, titles, and franchise names only when they matter to fan understanding. Don't clutter subtitles with unnecessary explanation.
8. If the host corrects herself, restarts a sentence, or hesitates, render it in natural spoken English. Don't add ellipses unless the transcript itself clearly shows an unfinished thought — when a thought continues into the next subtitle, prefer a comma or clean continuation.
9. When a joke or anecdote depends on phrasing, preserve the punchline or emotional beat even if sentence structure must change.
10. Assume the audience is familiar with anime, seiyuu, radio-show, and idol-fandom context. Translate clearly, but don't over-explain common fan references inside the subtitle itself.
11. Some shows follow a recurring structure with named corners/segments. When you recognize a cue phrase that signals a corner transition, adjust your tone to match that segment.
```

### What was cut

| Old | Why removed |
|---|---|
| #1 "spoken performance, not prose" (as a rule) | Folded into the opening paragraph |
| #2 "natural, conversational English" | Duplicates hardcoded translator rule #4 ("natural subtitle English over literal wording") |
| #3 "compact, subtitle-friendly" | Duplicates hardcoded translator rule #6 ("keep translations concise enough to work as readable subtitles") |
| #5 "Translate for listener experience, not dictionary" | Restates #2/#4 — covered by hardcoded #4 |
| Style constraints #1 "spoken English, not formal written" | Duplicates the cuts above |
| Style constraints #2 "avoid academic wording, passive" | Same |
| Style constraints #3 "emotional exclamations lively" | Sub-case of "preserve persona" (kept as #1) |
| Style constraints #4 "continuity across adjacent lines" | Duplicates hardcoded translator rules #2 and #3 (cross-line context, sentence distribution) |
| Style constraints #5 "no ellipses for filler" | Duplicates #8 (now), which is more explicit |

### What was kept

All radio-specific rules the generic translator prompt doesn't cover: fan-mail framing, quotative phrase handling, listener-vs-host line separation, filler-sound rendering, recurring bits, honorifics policy, self-correction handling, joke punchlines, fandom assumption, and corner-cue awareness.

---

## 2. `profiles/local/proseka/aftertalk.toml` (prompt strings only)

> Note: this file lives in `profiles/local/` and is gitignored. The edit applies locally to your runs but is not part of any commit.

### Before (~2,500 chars in the two prompt strings)

```
Program context:
ProSeka AfterTalk (プロセカアフタートーク) is a recurring livestream for the mobile rhythm game Project Sekai: Colorful Stage! feat. Hatsune Miku. Each edition covers one in-game event story: a single voice actor hosts, reads fan letters, and discusses the story while watching key scenes live. Streams air on YouTube after each event ends.

Casual register guidelines:
1. The host is reacting to story scenes, reading fan letters, sharing personal anecdotes, and commenting on their character's emotional arc. Translate with that context in mind.
2. When discussing in-game story scenes, preserve character names and event-specific terminology consistently.
3. This is a casual livestream, not a formal presentation. Use casual register: "Here we go!" not "Please, take a look"; "if you think I'm yapping too much" not "If you find me too noisy." Match the energy of a relaxed YouTube stream.
```

```
Style rules for ProSeka content:
1. When the host refers to themselves in third person by their name or character name (e.g., みのり, エナ), preserve this in English (e.g., "Minori will be talking" not "I'll be talking"). Third-person self-reference is a deliberate speech pattern, not an error.
2. Song titles must be translated exactly as they appear — do not paraphrase, add words, or change nouns (e.g., "painting" vs "drawing" matters). If unsure, transliterate rather than guess.
3. Preserve tsukkomi (self-directed comedic retorts). When the host catches themselves saying something funny or contradictory and calls it out (e.g., お前かい), translate the comedic self-awareness, not just the surface meaning.
4. Game card states 特訓前/特訓後 are "untrained/trained" — do NOT use ML-flavored variants like "pre-trained/post-trained", "pre-training/post-training", or "before/after training". Always use the bare adjective forms paired with "art" or "version" (e.g., "untrained art", "the trained version").
5. When the host mentions specific times (e.g., 5分), consider the livestream schedule context — these often refer to clock times (e.g., 10:05 PM) rather than durations.
6. Vocaloid character names (MEIKO, KAITO, MIKU, RIN, LEN, LUKA) are ALWAYS rendered in ALL CAPS in English. Watch for kanji renderings — 咲子 in Vocaloid context is MEIKO. Do not normalize these to title case.
7. Producer/artist names (Vocaloid scene composers like aqu3ra, MARETU, chouchou-P, Ayase) use their official Latin stylization, including numerals-for-letters (3=e), lowercase, and trailing -P. Never add Japanese honorifics (-san, -chan, -kun) to a producer name. If you cannot identify the canonical Latin spelling, leave the original katakana untransliterated rather than guessing a phonetic romanization.
8. Brand capitalization is non-negotiable: ProSeka (capital S, never "Proseka"), AfterTalk (capital T), AfterLive (capital L), SEKAI (all caps).
```

### After (~1,920 chars in the two prompt strings)

```
ProSeka AfterTalk (プロセカアフタートーク) is a recurring livestream for the mobile rhythm game Project Sekai: Colorful Stage! feat. Hatsune Miku. Each edition covers one in-game event story: a single voice actor hosts, reads fan letters, and discusses the story while watching key scenes live. Streams air on YouTube after each event ends.

Tone: relaxed YouTube livestream, not a formal presentation. Use casual register — "Here we go!" not "Please, take a look"; "if you think I'm yapping too much" not "If you find me too noisy."
```

```
ProSeka-specific style rules:
1. Preserve third-person self-reference. When the host refers to themselves by their own name or character name (みのり, エナ), keep it in English ("Minori will be talking", not "I'll be talking") — it's a deliberate speech pattern.
2. Song titles must be translated exactly as they appear — do not paraphrase, add words, or swap nouns ("painting" vs "drawing" matters). If unsure, transliterate rather than guess.
3. Preserve tsukkomi (self-directed comedic retorts). When the host catches themselves and calls it out (e.g., お前かい), translate the comedic self-awareness, not just the surface meaning.
4. 特訓前/特訓後 are "untrained/trained" (e.g., "untrained art", "the trained version"). Never use ML-flavored variants like "pre-trained/post-trained" or "before/after training".
5. Specific times like 5分 usually refer to clock times in the livestream schedule (e.g., 10:05 PM), not durations.
6. Vocaloid character names (MEIKO, KAITO, MIKU, RIN, LEN, LUKA) are ALWAYS ALL CAPS in English. 咲子 in Vocaloid context is MEIKO.
7. Producer/artist names (aqu3ra, MARETU, chouchou-P, Ayase) use their official Latin stylization — numerals-for-letters, lowercase, trailing -P. Never add Japanese honorifics. If you can't identify the canonical spelling, leave the original katakana rather than guessing a romanization.
8. Brand capitalization is non-negotiable: ProSeka (capital S), AfterTalk (capital T), AfterLive (capital L), SEKAI (all caps).
```

### What was cut

| Old | Why removed |
|---|---|
| "Program context:" header | Redundant — the paragraph below is self-explanatory |
| Casual register #1 ("host is reacting to story scenes, reading fan letters, sharing anecdotes...") | Duplicates cleaned `solo_seiyuu_radio.md` rule #2 + the program context paragraph already states the format |
| Casual register #2 ("preserve character names and event-specific terminology consistently") | Duplicates hardcoded translator rule #7 (catchphrase/term consistency); the glossary system enforces this mechanically |
| Rule #1 coda "Third-person self-reference is a deliberate speech pattern, not an error" | Merged into the rule itself: "— it's a deliberate speech pattern" |
| Rule #4 coda "Always use the bare adjective forms paired with 'art' or 'version'" | Replaced by inline example "(e.g., 'untrained art', 'the trained version')" |
| Rule #5 "When the host mentions specific times, consider the livestream schedule context" | Condensed to "Specific times like 5分 usually refer to clock times in the livestream schedule" |
| Rule #6 "Do not normalize these to title case" | Implied by "ALWAYS ALL CAPS" |
| Rule #7 "If you cannot identify the canonical Latin spelling, leave the original katakana untransliterated rather than guessing a phonetic romanization" | Tightened: "If you can't identify the canonical spelling, leave the original katakana rather than guessing a romanization" |

### What was kept

All 8 ProSeka-specific style rules — each defends against a real failure mode (third-person self-ref drift, paraphrased song titles, lost tsukkomi, ML-flavored "pre-trained" hallucinations, durations-vs-clock-times, Vocaloid name casing, producer-name romanization guessing, brand capitalization). The concrete style-anchor examples ("Here we go!" / "yapping too much") were preserved.

---

## What was NOT touched

| File | Reason |
|---|---|
| `autosub/pipeline/translate/translator.py` (13-rule hardcoded block) | Only file in this stack that lands upstream; deferred pending discussion. Some overlap candidates: #1+#2+#3 (cross-line context), #4+#6 (natural/concise). |
| `profiles/local/proseka/n25.toml` (unit context + translation notes) | All content is N25-specific factual context (character names, voice actors, personality notes, cross-unit relationships, game-specific terms). No general guidance to dedupe. |
| Glossary entries (built by `_build_glossary_prompt` in `cli.py`) | Already minimal — one line per term-pair. |
