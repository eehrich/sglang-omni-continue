# SPDX-License-Identifier: Apache-2.0
"""Pre-computed reference codes (``tts_params.ref_codes``) for MOSS-TTS Delay.

The decoder itself is shared with the Local Transformer and covered in
tests/unit_test/moss_tts_local/test_ref_codes.py. What is specific here is the
Delay path's wiring -- and one hazard that only exists because there are two
models: Delay runs ``n_vq`` 32 against Local's 12, so codes encoded for one
must not be accepted by the other.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_tts.request_builders import (
    _build_processor_message,
    resolve_moss_tts_ref_codes,
)
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.ref_codes import decode_reference_codes, encode_reference_codes

N_VQ = 32
N_VQ_LOCAL = 12
AUDIO_VOCAB_SIZE = 1024


def _codes(frames: int = 7, n_vq: int = N_VQ) -> torch.Tensor:
    return (torch.arange(frames * n_vq, dtype=torch.long) % AUDIO_VOCAB_SIZE).reshape(
        frames, n_vq
    )


class _RecordingProcessor:
    """Captures the reference the request builder hands to the processor."""

    model_config = type(
        "_FakeModelConfig",
        (),
        {"n_vq": N_VQ, "audio_vocab_size": AUDIO_VOCAB_SIZE},
    )()

    def __init__(self) -> None:
        self.reference = "unset"
        self.encoded_wavs = 0

    def build_user_message(self, **kwargs):
        self.reference = kwargs.get("reference")
        return dict(kwargs, role="user")

    def encode_audios_from_wav(self, wavs, sample_rate):
        self.encoded_wavs += 1
        return [_codes(3)]


def _payload(tts_params: dict, inputs: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id="req-codes",
        request=OmniRequest(
            inputs=inputs or {"text": "hallo"},
            params={},
            metadata={"task": "tts", "tts_params": tts_params},
        ),
        data={},
    )


def test_ref_codes_resolve_from_tts_params():
    codes = _codes(3)
    resolved = resolve_moss_tts_ref_codes(
        _payload({"ref_codes": encode_reference_codes(codes)}),
        processor=_RecordingProcessor(),
    )
    assert torch.equal(resolved, codes)


def test_ref_codes_resolve_from_reference_descriptor():
    codes = _codes(3)
    payload = _payload(
        {},
        inputs={
            "text": "hallo",
            "references": [{"ref_codes": encode_reference_codes(codes)}],
        },
    )
    assert torch.equal(
        resolve_moss_tts_ref_codes(payload, processor=_RecordingProcessor()), codes
    )


def test_absent_ref_codes_resolve_to_none():
    assert (
        resolve_moss_tts_ref_codes(
            _payload({"ref_audio": "/tmp/x.wav"}), processor=_RecordingProcessor()
        )
        is None
    )


def test_local_transformer_codes_are_rejected():
    """The reason the validator checks the second dimension at all.

    A 12-channel reference is a perfectly valid MOSS reference -- for the other
    model. Silently accepting it here would clone a voice from garbage rather
    than fail the request.
    """
    local_codes = encode_reference_codes(_codes(3, n_vq=N_VQ_LOCAL))
    with pytest.raises(ValueError):
        resolve_moss_tts_ref_codes(
            _payload({"ref_codes": local_codes}), processor=_RecordingProcessor()
        )


def test_codes_reach_the_processor_and_skip_the_encoder():
    """The whole point: the encode that codes replace must not still run."""
    processor = _RecordingProcessor()
    codes = _codes(5)
    state = MossTTSState(text="hallo", ref_audio=None, ref_text=None)

    _build_processor_message(processor, state, codes)

    assert processor.encoded_wavs == 0
    assert isinstance(processor.reference, list) and len(processor.reference) == 1
    assert torch.equal(processor.reference[0], codes)


def test_codes_win_over_ref_audio():
    """Both present means the same reference twice, one already encoded."""
    processor = _RecordingProcessor()
    codes = _codes(5)
    state = MossTTSState(text="hallo", ref_audio="/tmp/x.wav", ref_text=None)

    _build_processor_message(processor, state, codes)

    assert processor.encoded_wavs == 0
    assert torch.equal(processor.reference[0], codes)


def test_without_codes_the_audio_path_is_untouched():
    """Guards the existing request form: no codes, nothing changes."""
    processor = _RecordingProcessor()
    state = MossTTSState(text="hallo", ref_audio="/tmp/x.wav", ref_text=None)

    _build_processor_message(processor, state, None)

    assert processor.reference == ["/tmp/x.wav"]


def test_decoder_is_shared_and_parameterised_by_n_vq():
    """Same wire format, different model geometry -- one implementation."""
    for n_vq in (N_VQ_LOCAL, N_VQ):
        codes = _codes(4, n_vq=n_vq)
        packed = encode_reference_codes(codes)
        decoded = decode_reference_codes(
            packed, n_vq=n_vq, audio_vocab_size=AUDIO_VOCAB_SIZE
        )
        assert torch.equal(decoded, codes)
