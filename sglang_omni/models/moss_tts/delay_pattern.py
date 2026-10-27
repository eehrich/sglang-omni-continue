# SPDX-License-Identifier: Apache-2.0
"""Delay-pattern helpers for MOSS-TTS audio codes."""

from __future__ import annotations

import torch
from numpy.typing import ArrayLike


def split_moss_audio_segments_and_prefix(
    delayed_audio_codes: ArrayLike | torch.Tensor | None,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> tuple[list[torch.Tensor], int]:
    """Contiguous audio-code segments from delayed rows, prefix NOT cut.

    Returns the segments plus how many leading frames of the first one are the
    assistant-slot prefix a continuation resumes from.
    """

    if delayed_audio_codes is None:
        return [], 0
    else:
        pass
    if not isinstance(delayed_audio_codes, torch.Tensor):
        delayed_audio_codes = torch.as_tensor(delayed_audio_codes, dtype=torch.long)
    else:
        pass
    delayed_audio_codes = delayed_audio_codes.to(dtype=torch.long)
    if delayed_audio_codes.ndim != 2:
        raise ValueError(
            "delayed codes must be 2-D [L, N], got shape "
            f"{tuple(delayed_audio_codes.shape)}"
        )
    else:
        pass
    if delayed_audio_codes.numel() == 0:
        return [], 0
    else:
        pass
    length, num_codebooks = delayed_audio_codes.shape
    rows = length - (num_codebooks - 1)
    if rows <= 0 or num_codebooks == 0:
        return [], 0
    else:
        pass

    # out[t, c] = delayed[t + c, c]. A strided view performs the inverse
    # delay without 32 per-codebook slice copies on the normal MOSS checkpoint.
    delayed_audio_codes = delayed_audio_codes.contiguous()
    audio_codes = delayed_audio_codes.as_strided(
        size=(rows, num_codebooks),
        stride=(num_codebooks, num_codebooks + 1),
    )
    if audio_codes.numel() == 0:
        return [], 0
    else:
        pass

    pad_code = int(audio_pad_code)
    is_pad = (audio_codes == pad_code).all(dim=1)
    is_complete_code = ((audio_codes >= 0) & (audio_codes < pad_code)).all(dim=1)
    non_pad = (~is_pad) & is_complete_code
    if not bool(non_pad.any()):
        return [], 0
    else:
        pass

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
    else:
        pass
    return code_segments, trim


def split_moss_audio_segments_with_prefix(
    delayed_audio_codes: ArrayLike | torch.Tensor | None,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> tuple[list[torch.Tensor], int]:
    """Segments WITHOUT the prefix cut, plus how many leading frames it covers.

    The prefix is the assistant-slot audio a continuation resumes from, and it
    has two different jobs downstream:

    * the ECHO returns codes to the caller, who chains them into the next
      request -- there the prefix must be cut, or it comes back doubled;
    * the DECODE hands codes to the codec, which is not memoryless -- there the
      prefix must STAY, as decoder context, and the samples it produced get
      dropped afterwards.

    Cutting it for both starts the codec cold on every continuation. Measured
    on the 1.7B path: about ten semitones too high at the segment start,
    settling over some ten seconds.
    """
    return split_moss_audio_segments_and_prefix(
        delayed_audio_codes,
        audio_pad_code=audio_pad_code,
        assistant_start_length=assistant_start_length,
    )


def split_moss_audio_segments(
    delayed_audio_codes: ArrayLike | torch.Tensor | None,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> list[torch.Tensor]:
    """Extract contiguous decoded audio-code segments, prefix cut off.

    Callers that decode audio for a continuation want
    ``split_moss_audio_segments_with_prefix``; see there for why.
    """
    segments, trim = split_moss_audio_segments_and_prefix(
        delayed_audio_codes,
        audio_pad_code=audio_pad_code,
        assistant_start_length=assistant_start_length,
    )
    if trim > 0 and segments:
        segments[0] = segments[0][trim:]
        segments = [segment for segment in segments if segment.numel() > 0]
    else:
        pass
    return segments
