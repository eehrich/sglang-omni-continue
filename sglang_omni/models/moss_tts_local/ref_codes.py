# SPDX-License-Identifier: Apache-2.0
"""Pre-computed reference codes for MOSS-TTS Local.

The clone reference normally arrives as audio and is run through the ~1B-param
MOSS-Audio-Tokenizer-v2 encoder on every request. That encode does not
amortize across a batch the way decode does, so it is the throughput ceiling
for a workload whose reference differs per request (a re-anchored sliding
window): measured on a 5090, a 70 s window costs ~11% of a single request but
~59% of a request at concurrency 16.

The encode is avoidable. The HF processor's ``_resolve_audio_items`` accepts a
``torch.Tensor`` of shape ``[T, n_vq]`` anywhere it accepts a path, and takes
it verbatim -- exactly the layout ``MossTTSLocalModel.generate()`` emits. So a
caller that already knows its reference (a fixed narrator voice, or a window
it generated itself) can encode ONCE, offline, and hand the codes in.

Wire format (``metadata.tts_params.ref_codes``), primary form::

    {"data": "<base64>", "shape": [T, n_vq], "dtype": "int16"}

``data`` is base64 of the C-contiguous, little-endian, row-major ``[T, n_vq]``
array. base64-int16 was chosen over a nested JSON int array because it is
~2x smaller on the wire and O(n) memcpy to parse instead of allocating T*n_vq
Python ints: a 70 s reference is 875 x 12 = 10500 codes, i.e. ~28 KB base64
versus ~50 KB of JSON text -- and against the ~9 MB base64 WAV it replaces,
either form is a rounding error, so the tie-breaker is parse cost inside the
preprocessing worker. int16 is lossless here because every code is bounded by
``audio_vocab_size`` (1024).

A nested list of ints (``[[c, ...], ...]``) is also accepted for hand-written
requests and tests; it is not the production form.
"""

from __future__ import annotations

import base64
import binascii
import sys
from typing import Any

import torch

from sglang_omni.proto.admin import ADMIN_MOSS_ENCODE_REFERENCE

REF_CODES_PARAM = "ref_codes"

# Admin control-plane action that encodes audio into this wire form using the
# codec the preprocessing stage already holds. Defined in proto.admin (with the
# other actions) so the serve layer can name it without importing torch; see
# ``stages.build_reference_encode_admin_handler`` for the handler and
# ``serve/openai_api.py`` for the HTTP route (POST /moss/encode_reference).
ADMIN_ENCODE_REFERENCE = ADMIN_MOSS_ENCODE_REFERENCE

# The codec runs at 12.5 frames/s and stages._MAX_REFERENCE_SECONDS caps an
# audio reference at 100 s. Mirror that cap in frames so a code payload cannot
# smuggle in a longer prefix than the audio path would have allowed (importing
# the constant would be circular: stages imports this module's caller).
_CODEC_FRAMES_PER_SECOND = 12.5
CODEC_FRAMES_PER_SECOND = _CODEC_FRAMES_PER_SECOND
_MAX_REFERENCE_SECONDS = 100.0
MAX_REFERENCE_CODE_FRAMES = int(_MAX_REFERENCE_SECONDS * _CODEC_FRAMES_PER_SECOND)

# Unsigned widths map onto the same-width signed torch dtype: every code is
# < audio_vocab_size (1024), so the sign bit is never set and the
# reinterpretation is byte-exact.
_DTYPES: dict[str, torch.dtype] = {
    "int16": torch.int16,
    "uint16": torch.int16,
    "int32": torch.int32,
    "uint32": torch.int32,
}
_ITEMSIZES: dict[str, int] = {"int16": 2, "uint16": 2, "int32": 4, "uint32": 4}
_DEFAULT_DTYPE = "int16"


def _fail(field: str, message: str) -> None:
    raise ValueError(f"MOSS-TTS Local {field}: {message}")


def _validate_tensor(
    codes: torch.Tensor,
    *,
    n_vq: int,
    audio_vocab_size: int,
    field: str,
) -> torch.Tensor:
    if codes.ndim != 2:
        _fail(field, f"expected a 2-D [T, {n_vq}] array, got ndim={codes.ndim}")
    frames, channels = int(codes.shape[0]), int(codes.shape[1])
    if channels != n_vq:
        _fail(
            field,
            f"expected shape [T, {n_vq}] (row-major, one row per codec frame), "
            f"got [{frames}, {channels}]",
        )
    if frames < 1:
        _fail(field, "reference codes must contain at least one frame")
    if frames > MAX_REFERENCE_CODE_FRAMES:
        _fail(
            field,
            f"reference is {frames} frames (~{frames / _CODEC_FRAMES_PER_SECOND:.1f}s), "
            f"limit is {MAX_REFERENCE_CODE_FRAMES} frames "
            f"(~{_MAX_REFERENCE_SECONDS:.0f}s)",
        )
    codes = codes.to(dtype=torch.long)
    low = int(codes.min())
    high = int(codes.max())
    # audio_vocab_size itself is the pad code; a reference must not carry pads.
    if low < 0 or high >= audio_vocab_size:
        _fail(
            field,
            f"codes must lie in [0, {audio_vocab_size - 1}] "
            f"(audio_vocab_size={audio_vocab_size}), got [{low}, {high}]",
        )
    return codes.contiguous()


def _decode_packed(payload: dict[str, Any], *, field: str) -> torch.Tensor:
    data = payload.get("data")
    if data is None:
        _fail(field, "packed form requires 'data' (base64) and 'shape'")
    if not isinstance(data, str):
        _fail(field, f"'data' must be a base64 string, got {type(data).__name__}")

    dtype_name = str(payload.get("dtype") or _DEFAULT_DTYPE).lower()
    if dtype_name not in _DTYPES:
        _fail(
            field,
            f"unsupported dtype {dtype_name!r}; expected one of "
            f"{sorted(_DTYPES)}",
        )

    shape = payload.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        _fail(field, f"'shape' must be [T, n_vq], got {shape!r}")
    try:
        frames, channels = int(shape[0]), int(shape[1])
    except (TypeError, ValueError):
        _fail(field, f"'shape' entries must be integers, got {shape!r}")

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        _fail(field, f"'data' is not valid base64 ({exc})")

    expected = frames * channels * _ITEMSIZES[dtype_name]
    if len(raw) != expected:
        _fail(
            field,
            f"decoded {len(raw)} bytes but shape {[frames, channels]} at "
            f"{dtype_name} needs {expected}",
        )
    if sys.byteorder != "little":
        _fail(field, "packed reference codes are little-endian; host is big-endian")

    # bytearray() so torch owns a writable copy (torch.frombuffer refuses to
    # alias an immutable bytes object without a warning).
    flat = torch.frombuffer(bytearray(raw), dtype=_DTYPES[dtype_name])
    return flat.reshape(frames, channels)


def decode_reference_codes(
    raw: Any,
    *,
    n_vq: int,
    audio_vocab_size: int,
    field: str = REF_CODES_PARAM,
) -> torch.Tensor:
    """Turn a wire ``ref_codes`` value into a validated ``[T, n_vq]`` long tensor.

    Accepts the packed dict form, a nested int list, or an already-built
    tensor (in-process callers). Raises ``ValueError`` with an actionable
    message on any shape/range/dtype mismatch -- a bad code would otherwise
    index outside the audio embedding table and produce garbage audio rather
    than an error.
    """
    if isinstance(raw, torch.Tensor):
        codes = raw
    elif isinstance(raw, dict):
        codes = _decode_packed(raw, field=field)
    elif isinstance(raw, (list, tuple)):
        if not raw:
            _fail(field, "reference codes must contain at least one frame")
        try:
            codes = torch.tensor(raw, dtype=torch.long)
        except (TypeError, ValueError) as exc:
            _fail(field, f"nested list form must be [[int, ...], ...] ({exc})")
    else:
        _fail(
            field,
            "must be a packed {'data': base64, 'shape': [T, n_vq]} object or a "
            f"nested int list, got {type(raw).__name__}",
        )
    return _validate_tensor(
        codes, n_vq=n_vq, audio_vocab_size=audio_vocab_size, field=field
    )


def encode_reference_codes(codes: torch.Tensor) -> dict[str, Any]:
    """Pack a ``[T, n_vq]`` code tensor into the wire form (offline producers)."""
    if codes.ndim != 2:
        raise ValueError(f"expected a 2-D [T, n_vq] tensor, got ndim={codes.ndim}")
    packed = codes.to(dtype=torch.int16).contiguous()
    return {
        "data": base64.b64encode(packed.numpy().tobytes()).decode("ascii"),
        "shape": [int(packed.shape[0]), int(packed.shape[1])],
        "dtype": "int16",
    }
