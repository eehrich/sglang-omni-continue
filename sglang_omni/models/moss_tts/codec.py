# SPDX-License-Identifier: Apache-2.0
"""Delay-pattern helpers for MOSS-TTS audio codes."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.utils.codec_delay import reverse_delay_pattern


def split_moss_audio_segments_with_prefix(
    delayed_audio_codes: Any,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> tuple[list[torch.Tensor], int]:
    """Segments WITHOUT the prefix cut, plus how many leading frames it covers.

    The prefix is the assistant-slot audio a continuation resumes from, and it
    has two different jobs downstream, which is why splitting the "compute it"
    from the "apply it" matters:

    * the ECHO returns codes to the caller, who chains them into the next
      request -- there the prefix must be cut, or it comes back doubled;
    * the DECODE hands codes to the codec, which is not memoryless -- there the
      prefix must STAY, as decoder context, and the samples it produced get
      dropped afterwards.

    Cutting it here for both, as this module did, starts the codec cold on
    every continuation. Measured on the 1.7B path, where the same defect sat
    one layer later: about ten semitones too high at the segment start,
    settling over some ten seconds.
    """
    segments, trim = _split(
        delayed_audio_codes,
        audio_pad_code=audio_pad_code,
        assistant_start_length=assistant_start_length,
    )
    return segments, trim


def split_moss_audio_segments(
    delayed_audio_codes: Any,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> list[torch.Tensor]:
    """Segments WITH the prefix cut off the first one -- the echo's view.

    Callers that decode audio want ``split_moss_audio_segments_with_prefix``
    instead; see there for why the two differ.
    """
    segments, trim = _split(
        delayed_audio_codes,
        audio_pad_code=audio_pad_code,
        assistant_start_length=assistant_start_length,
    )
    if trim > 0 and segments:
        segments[0] = segments[0][trim:]
        segments = [segment for segment in segments if segment.numel() > 0]
    return segments


def _split(
    delayed_audio_codes: Any,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> tuple[list[torch.Tensor], int]:
    """Extract contiguous decoded audio-code segments from delayed rows."""

    if delayed_audio_codes is None:
        return [], 0
    if not isinstance(delayed_audio_codes, torch.Tensor):
        delayed_audio_codes = torch.as_tensor(delayed_audio_codes, dtype=torch.long)
    delayed_audio_codes = delayed_audio_codes.to(dtype=torch.long)
    if delayed_audio_codes.numel() == 0:
        return [], 0

    audio_codes = reverse_delay_pattern(delayed_audio_codes, allow_short=True)
    if audio_codes.numel() == 0:
        return [], 0

    pad_code = int(audio_pad_code)
    is_pad = (audio_codes == pad_code).all(dim=1)
    is_complete_code = ((audio_codes >= 0) & (audio_codes < pad_code)).all(dim=1)
    non_pad = (~is_pad) & is_complete_code
    if not bool(non_pad.any()):
        return [], 0

    idx = torch.nonzero(non_pad, as_tuple=False).squeeze(1)
    break_points = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    if break_points.numel() == 0:
        segments = [idx]
    else:
        segments = list(torch.tensor_split(idx, break_points.cpu().tolist()))

    code_segments = [audio_codes[segment].contiguous() for segment in segments]
    trim = 0
    if assistant_start_length > 0 and code_segments:
        # The prefix sits in the FIRST segment by construction: it precedes
        # every generated frame and a pad gap cannot open inside it.
        trim = min(int(assistant_start_length), int(code_segments[0].shape[0]))
    return code_segments, trim
