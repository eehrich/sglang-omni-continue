# MOSS-TTS Delay (8B): windowed references raise the pitch on sglang-omni

> **Resolved.** The cause was the grouping of the audio repetition penalty:
> sglang applied it per codebook, the reference implementation applies one
> shared token set across codebooks 1..n-1. See "The cause" at the end; the
> investigation is kept because most of it is negative results that are worth
> not repeating.

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
- **The delay stagger of the fed-back codes.** A deterministic decode (top_k
  1) shows the two engines emit the same channel-0 stream but in different
  shapes: HF returns `[T + n_vq - 1, n_vq]` rows where channel `c` is shifted
  by `c` frames and the inactive channels hold `audio_pad`, while this port
  reverses that to the plain `[T, n_vq]` an encoder produces — the 144 vs 113
  frame counts differ by exactly `n_vq - 1`. Feeding the stagger back
  verbatim (pad preserved, validator opened to accept it) changes nothing:
  117.6 Hz staggered vs 115.1 plain, control 99.7. An earlier version of this
  test clipped the codes to `[0, 1023]` and so destroyed the pad markers; its
  result was meaningless.
- **`audio_length` seeded from the prompt.** `_initialize_generation_state`
  treats a clone prompt as a continuation (it also ends on the assistant gen
  slot), so `audio_length` starts at the reference length and
  `pre_audio = audio_lengths > channel_idx` is saturated before the first
  generated frame. Forcing it to 0 for closed reference blocks made the
  effect worse (137.9), so this is not the cause either.

## Where it likely sits

Something in the Delay engine's own emission differs from
`model.generate()`'s in a way that is inaudible in the segment itself but
that a later clone reads as part of the voice.

It survives a full audio round trip: re-encoding the generated WAV through
the codec, so that the window carries ordinary encoder codes rather than the
echo, still leaves the pitch at 110.7 Hz against 94.7 for real speech. So it
is carried by the waveform, not by the code representation — which also rules
the wire format out on its own terms.

Candidates not yet excluded: the delay bookkeeping in
`model_runner._sample_rows` (`pre_audio` / `post_audio` masks, the
`delayed`/`audio_lengths` update), the channel-splitting heuristic in
`sglang_model._prepare_multi_modal_inputs`, which infers the layout from
`total_tokens % channels`, and the vocoder stage's own reversal
(`split_moss_audio_segments`) against the processor's `_parse_audio_codes` —
the two agree with each other (correlation 1.000) but have not been checked
against HF's parse of the same rows.

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

## The cause

Per-codebook statistics over two predecessors generated from the same base,
text and seed showed sglang's codes carrying about **0.34 bits more entropy in
every one of the 32 codebooks** — uniformly, channel 0 least. Same nominal
temperature, top-p and top-k, so something was flattening the distribution
less than the reference does.

It is the audio repetition penalty. `generate()` hands
`apply_repetition_penalty_delay_pattern` logits of shape `[N, V]` for the
audio heads, because the mask indexing in the loop flattens batch and codebook
together — so its delay-pattern branch never runs and the 2-D branch takes
`prev_tokens.reshape(-1)`. Codebook 0 is penalised against its own history,
while codebooks 1..n-1 share **one** token set: the union of their histories,
applied identically to each. sglang grouped it per codebook, which is the
natural reading of the function's name but not what the caller reaches.

Penalising a union instead of a column pushes far more of the vocabulary down,
and since most logits sit below the maximum that sharpens the distribution.
The gap grows with reference length, because a longer window widens the union
— which is exactly why single clones were fine and a re-anchored window was
not.

With the grouping matched (`model_runner._apply_audio_repetition_penalty`):

| | before | after |
|---|---|---|
| 39 s window, generated content | 120.3 Hz | **103.9** |
| 39 s window, real content | 94.7 | 100.6 |
| six-step chain | 115.5 | **99.0** |
| six-step clone control | 94.2 | 96.7 |

The asymmetry between generated and real content in the window — 25 Hz before,
3 Hz after — is the part that matters: those two should behave alike, and now
they do. Chain against clone falls from +3.5 semitones to +0.4, against 0 for
the HF reference, which is inside the per-take spread of these measurements.

## Why it matters

For single-shot cloning the Delay model on sglang-omni is fine — this only
shows up when its own output is fed back as conditioning, which is what
long-form narration does at every segment boundary.
