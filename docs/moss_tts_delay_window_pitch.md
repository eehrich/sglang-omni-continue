# MOSS-TTS Delay (8B): windowed references raise the pitch on sglang-omni

Serving MOSS-TTS-v1.5 (Delay, `n_vq` 32) through sglang-omni and conditioning
each segment on a re-anchored sliding window — base voice plus the previously
generated segment, the shape long-form narration uses — raises the fundamental
frequency by roughly four semitones and holds it there. The same window, the
same base reference, the same texts and seeds through `transformers`
`model.generate()` do not move it.

Measured on a 5090 + 3090 split, `MOSS-TTS-v1.5`, German, sampling at the
model's own defaults (`audio_temperature` 1.7, `audio_top_p` 0.80,
`audio_top_k` 25, `audio_repetition_penalty` 1.05, `text_temperature` 1.5,
`text_top_p` 1.0, `text_top_k` 50). F0 is the median over autocorrelation
estimates of voiced frames.

## The observation

Six-step chain, real narration texts, three seeds per arm. Every step
conditions on `concat(base_reference, previous_generated_segment)`; the
control conditions on the base reference alone.

| six-step chain, 3 seeds | median F0 |
|---|---|
| sglang, window | **115.5 Hz** |
| sglang, clone (base only) | 94.2 Hz |
| HF `generate()`, window | **92 Hz** |
| HF `generate()`, clone | 92 Hz |

The rise appears at the first windowed step and plateaus — it does not creep
up over the chain (`r` with step index +0.04), and the segments already start
high (first third 102–139 Hz), so it is the conditioning rather than
something accumulating during decode.

## What isolates it

Holding the window length fixed at 39 s and swapping only what fills the
second half:

| window content | median F0 |
|---|---|
| base only (13 s) | 93.6 |
| base + **generated** audio | **120.3** |
| base + **real** audio (base repeated) | 94.7 |

So it is not length: a window of the same size filled with the original
recording is fine. The generated predecessor itself measures 90.9 Hz and is
flat across its own quarters (92.5 / 90.9 / 92.0 / 88.9), i.e. it neither
sounds high nor ends high.

Crossing the two engines pins the side it comes from:

| | median F0 |
|---|---|
| HF window + HF's own predecessor | 88.4 |
| HF window + **sglang's** predecessor | **102.6** |

sglang's generated codes raise the pitch in the HF engine too. The defect
travels with what sglang produces, not with how it consumes a reference.

## Ruled out

Each with at least three seeds, because per-take F0 spread on identical
input is 89–113 Hz and single generations cannot separate anything smaller
than about a semitone.

- **Reference transport.** `ref_codes` and `ref_audio` give identical medians
  (95.2 Hz at 13 s, 99.4 Hz at 39 s), seed for seed.
- **The echo itself.** Codes returned via `return_codes` decode back to
  exactly the audio of the same request: waveform correlation 1.000, same
  duration, same F0.
- **Reference length.** Longer reference *and* longer text both trend
  slightly downward: 10 s/short 103.9, 10 s/long 93.6, 39 s/short 93.0,
  39 s/long 85.6.
- **The concatenation seam** between separately encoded blocks: 99.4 in one
  piece vs 101.9 as three blocks.
- **Reference format**: 48 kHz stereo, 48 kHz mono and 24 kHz mono behave the
  same.
- **Loudness**: base recording −20.0 dB mean, generated predecessor −19.6 dB.
- **Delay layout of the fed-back codes.** Re-applying the delay pattern
  before conditioning makes it worse, not better (128.0 vs 115.1; control
  99.7), so the plain `[T, n_vq]` rows are the right form.
- **Trimming the head** of the predecessor (its delay ramp-in region): 111.1
  without the first 32 frames, 108.8 without the first 64 — still elevated.
- **`audio_length` seeded from the prompt.** `_initialize_generation_state`
  treats a clone prompt as a continuation (it also ends on the assistant gen
  slot), so `audio_length` starts at the reference length and
  `pre_audio = audio_lengths > channel_idx` is saturated before the first
  generated frame. Forcing it to 0 for closed reference blocks made the
  effect worse (137.9), so this is not the cause either.

## Where it likely sits

Something in the Delay engine's own emission differs from
`model.generate()`'s in a way that is inaudible in the segment itself but
that a later clone reads as part of the voice. The audio is fine; the codes
carry it. Candidates not yet excluded: the delay bookkeeping in
`model_runner._sample_rows` (`pre_audio` / `post_audio` masks, the
`delayed`/`audio_lengths` update), and the channel-splitting heuristic in
`sglang_model._prepare_multi_modal_inputs`, which infers the layout from
`total_tokens % channels`.

## A mitigation that works today

The rise scales with how much generated audio the window carries, so capping
the predecessor to a short tail buys most of it back. Six-step chains, three
seeds, same base anchor, only the predecessor portion trimmed:

| predecessor kept in the window | chain median F0 |
|---|---|
| all of it (20–26 s) | 115.5 |
| last 10 s | 104.6 |
| **last 4 s** | **100.0** |
| none (clone control) | 93.3 |

Four seconds still carries the prosodic context the window exists for and
lands within the spread of the clone control, so it is a usable setting until
the engine side is fixed. Ten seconds is not enough of a cap: in a chain the
residue still compounds.

## Why it matters

For single-shot cloning the Delay model on sglang-omni is fine — this only
shows up when its own output is fed back as conditioning, which is what
long-form narration does at every segment boundary.
