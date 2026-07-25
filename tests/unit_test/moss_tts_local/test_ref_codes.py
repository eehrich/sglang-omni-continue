# SPDX-License-Identifier: Apache-2.0
"""Pre-computed reference codes (``tts_params.ref_codes``) for MOSS-TTS Local."""

from __future__ import annotations

import base64

import pytest
import torch

from sglang_omni.models.moss_tts_local.ref_codes import (
    MAX_REFERENCE_CODE_FRAMES,
    decode_reference_codes,
    encode_reference_codes,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    clear_moss_tts_local_preprocessing_context,
    pop_prepared_moss_tts_local_request,
    preprocess_moss_tts_local_payload,
    resolve_moss_tts_local_ref_codes,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.pipeline.control_plane import serialize_message
from sglang_omni.proto import OmniRequest, StagePayload

N_VQ = 12
AUDIO_VOCAB_SIZE = 1024


def _codes(frames: int = 7) -> torch.Tensor:
    return (torch.arange(frames * N_VQ, dtype=torch.long) % AUDIO_VOCAB_SIZE).reshape(
        frames, N_VQ
    )


def _decode(raw):
    return decode_reference_codes(
        raw, n_vq=N_VQ, audio_vocab_size=AUDIO_VOCAB_SIZE
    )


def test_packed_round_trip_is_lossless():
    codes = _codes()
    packed = encode_reference_codes(codes)
    assert packed["shape"] == [7, N_VQ]
    assert packed["dtype"] == "int16"
    decoded = _decode(packed)
    assert decoded.dtype == torch.long
    assert torch.equal(decoded, codes)


def test_nested_list_form_is_accepted():
    codes = _codes(3)
    assert torch.equal(_decode(codes.tolist()), codes)


def test_tensor_passthrough_is_validated_and_cast():
    codes = _codes(3).to(torch.int16)
    decoded = _decode(codes)
    assert decoded.dtype == torch.long
    assert torch.equal(decoded, codes.long())


@pytest.mark.parametrize("dtype", ["int16", "uint16", "int32", "uint32"])
def test_supported_dtypes(dtype):
    codes = _codes(4)
    torch_dtype = torch.int16 if dtype.endswith("16") else torch.int32
    raw = codes.to(torch_dtype).contiguous().numpy().tobytes()
    packed = {
        "data": base64.b64encode(raw).decode("ascii"),
        "shape": [4, N_VQ],
        "dtype": dtype,
    }
    assert torch.equal(_decode(packed), codes)


def test_wrong_n_vq_is_rejected():
    with pytest.raises(ValueError, match=f"expected shape \\[T, {N_VQ}\\]"):
        _decode(encode_reference_codes(torch.zeros((5, N_VQ - 1), dtype=torch.long)))


def test_transposed_payload_is_rejected():
    # [n_vq, T] is the codec's own pre-transpose layout and the most likely
    # producer mistake; it must fail loudly rather than clone a garbage voice.
    with pytest.raises(ValueError, match="expected shape"):
        _decode(encode_reference_codes(_codes(40).transpose(0, 1).contiguous()))


def test_byte_length_mismatch_is_rejected():
    packed = encode_reference_codes(_codes(6))
    packed["shape"] = [5, N_VQ]
    with pytest.raises(ValueError, match="needs"):
        _decode(packed)


def test_out_of_range_code_is_rejected():
    codes = _codes(3)
    codes[1, 2] = AUDIO_VOCAB_SIZE  # the pad code, never valid in a reference
    with pytest.raises(ValueError, match="codes must lie in"):
        _decode(codes)


def test_reference_length_cap():
    too_long = torch.zeros((MAX_REFERENCE_CODE_FRAMES + 1, N_VQ), dtype=torch.long)
    with pytest.raises(ValueError, match="limit is"):
        _decode(too_long)


def test_bad_base64_is_rejected():
    with pytest.raises(ValueError, match="not valid base64"):
        _decode({"data": "not base64!!", "shape": [1, N_VQ]})


def test_unsupported_type_is_rejected():
    with pytest.raises(ValueError, match="must be a packed"):
        _decode(42)


class _RecordingProcessor:
    """Captures the reference the request builder hands to the processor."""

    model_config = type(
        "_FakeModelConfig",
        (),
        {"n_vq": N_VQ, "audio_vocab_size": AUDIO_VOCAB_SIZE},
    )()

    def __init__(self) -> None:
        self.reference = "unset"

    def build_user_message(self, **kwargs):
        self.reference = kwargs.get("reference")
        return dict(kwargs, role="user")

    def __call__(self, conversations, mode):
        rows = torch.full((1, 4, N_VQ + 1), AUDIO_VOCAB_SIZE, dtype=torch.long)
        rows[0, :, 0] = torch.arange(4)
        return {"input_ids": rows}


class _ExplodingEncoder:
    """Any call means a codec encode was about to run -- the thing we removed."""

    def encode(self, path):  # pragma: no cover - must not be reached
        raise AssertionError("reference encoder must not run when ref_codes is given")

    def encode_data_uri(self, ref_audio):  # pragma: no cover - must not be reached
        raise AssertionError("reference encoder must not run when ref_codes is given")


def _payload(tts_params: dict) -> StagePayload:
    return StagePayload(
        request_id="req-codes",
        request=OmniRequest(
            inputs={"text": "hallo welt"},
            params={},
            metadata={"task": "tts", "tts_params": tts_params},
        ),
        data={},
    )


def test_ref_codes_resolve_from_tts_params():
    codes = _codes(3)
    resolved = resolve_moss_tts_local_ref_codes(
        _payload({"ref_codes": encode_reference_codes(codes)}),
        processor=_RecordingProcessor(),
    )
    assert torch.equal(resolved, codes)


def test_ref_codes_resolve_from_reference_descriptor():
    codes = _codes(3)
    payload = StagePayload(
        request_id="req-codes",
        request=OmniRequest(
            inputs={
                "text": "hallo",
                "references": [{"ref_codes": encode_reference_codes(codes)}],
            },
            params={},
            metadata={"task": "tts", "tts_params": {}},
        ),
        data={},
    )
    resolved = resolve_moss_tts_local_ref_codes(
        payload, processor=_RecordingProcessor()
    )
    assert torch.equal(resolved, codes)


def test_absent_ref_codes_resolve_to_none():
    assert (
        resolve_moss_tts_local_ref_codes(
            _payload({"ref_audio": "/tmp/x.wav"}), processor=_RecordingProcessor()
        )
        is None
    )


def test_prepared_state_stays_msgpack_serializable():
    """The terminal stage hop msgpacks the state dict; a tensor there is fatal.

    A code tensor parked on MossTTSLocalState killed the vocoder process with
    "can not serialize 'Tensor' object", so the codes must never reach the
    wire form of the state.
    """
    processor = _RecordingProcessor()
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=_ExplodingEncoder()
    )
    try:
        payload = preprocess_moss_tts_local_payload(
            _payload({"ref_codes": encode_reference_codes(_codes(5))})
        )
        prepared = pop_prepared_moss_tts_local_request(payload)
    finally:
        clear_moss_tts_local_preprocessing_context()
    assert prepared is not None
    wire = prepared.state.to_dict()
    assert not any(isinstance(value, torch.Tensor) for value in wire.values())
    serialize_message(_FakeCompleteMessage(wire))


class _FakeCompleteMessage:
    """Minimal stand-in for the control-plane message that carries state data."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return {"kind": "complete", "request_id": "req-codes", "data": self._data}


def test_preprocess_routes_codes_and_skips_the_encoder():
    codes = _codes(9)
    processor = _RecordingProcessor()
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=_ExplodingEncoder()
    )
    try:
        payload = preprocess_moss_tts_local_payload(
            _payload(
                {
                    # ref_audio present too: codes must win, no encode.
                    "ref_audio": "data:audio/wav;base64,AAAA",
                    "ref_codes": encode_reference_codes(codes),
                }
            )
        )
        prepared = pop_prepared_moss_tts_local_request(payload)
        assert prepared is not None
    finally:
        clear_moss_tts_local_preprocessing_context()

    assert isinstance(processor.reference, list) and len(processor.reference) == 1
    assert torch.equal(processor.reference[0], codes)


def test_preprocess_rejects_malformed_codes():
    processor = _RecordingProcessor()
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=_ExplodingEncoder()
    )
    try:
        with pytest.raises(ValueError, match="expected shape"):
            preprocess_moss_tts_local_payload(
                _payload({"ref_codes": [[1, 2, 3]]}),
            )
    finally:
        clear_moss_tts_local_preprocessing_context()
