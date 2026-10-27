# SPDX-License-Identifier: Apache-2.0
"""Pre-computed reference codes (``tts_params.ref_codes``) for MOSS-TTS Local."""

from __future__ import annotations

import base64

import pytest
import torch

from sglang_omni.models.moss_tts.reference_encoder import (
    build_reference_encode_admin_handler,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    MossTTSLocalPreparedRequest,
    clear_moss_tts_local_preprocessing_context,
    preprocess_moss_tts_local_payload,
    resolve_moss_ref_codes,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.pipeline.control_plane import serialize_message
from sglang_omni.proto import (
    ADMIN_MOSS_ENCODE_REFERENCE,
    AdminResult,
    AdminResultMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.utils.ref_codes import (
    MAX_REFERENCE_CODE_FRAMES,
    decode_reference_codes,
    encode_reference_codes,
)

N_VQ = 12
AUDIO_VOCAB_SIZE = 1024
PREPARED_MARKER = "_moss_tts_local_prepared_request"


def make_codes(frames: int = 7) -> torch.Tensor:
    return (torch.arange(frames * N_VQ, dtype=torch.long) % AUDIO_VOCAB_SIZE).reshape(
        frames, N_VQ
    )


def decode(raw):
    return decode_reference_codes(raw, n_vq=N_VQ, audio_vocab_size=AUDIO_VOCAB_SIZE)


def test_packed_round_trip_is_lossless():
    codes = make_codes()
    packed = encode_reference_codes(codes)
    assert packed["shape"] == [7, N_VQ]
    assert packed["dtype"] == "int16"
    decoded = decode(packed)
    assert decoded.dtype == torch.long
    assert torch.equal(decoded, codes)


def test_nested_list_form_is_accepted():
    codes = make_codes(3)
    assert torch.equal(decode(codes.tolist()), codes)


def test_tensor_passthrough_is_validated_and_cast():
    codes = make_codes(3).to(torch.int16)
    decoded = decode(codes)
    assert decoded.dtype == torch.long
    assert torch.equal(decoded, codes.long())


@pytest.mark.parametrize("dtype", ["int16", "uint16", "int32", "uint32"])
def test_supported_dtypes(dtype):
    codes = make_codes(4)
    torch_dtype = torch.int16 if dtype.endswith("16") else torch.int32
    raw = codes.to(torch_dtype).contiguous().numpy().tobytes()
    packed = {
        "data": base64.b64encode(raw).decode("ascii"),
        "shape": [4, N_VQ],
        "dtype": dtype,
    }
    assert torch.equal(decode(packed), codes)


def test_wrong_n_vq_is_rejected():
    with pytest.raises(ValueError, match=f"expected shape \\[T, {N_VQ}\\]"):
        decode(encode_reference_codes(torch.zeros((5, N_VQ - 1), dtype=torch.long)))


def test_transposed_payload_is_rejected():
    # [n_vq, T] is the codec's own pre-transpose layout and the most likely
    # producer mistake; it must fail loudly rather than clone a garbage voice.
    with pytest.raises(ValueError, match="expected shape"):
        decode(encode_reference_codes(make_codes(40).transpose(0, 1).contiguous()))


def test_byte_length_mismatch_is_rejected():
    packed = encode_reference_codes(make_codes(6))
    packed["shape"] = [5, N_VQ]
    with pytest.raises(ValueError, match="needs"):
        decode(packed)


def test_out_of_range_code_is_rejected():
    codes = make_codes(3)
    codes[1, 2] = AUDIO_VOCAB_SIZE  # the pad code, never valid in a reference
    with pytest.raises(ValueError, match="codes must lie in"):
        decode(codes)


def test_reference_length_cap():
    too_long = torch.zeros((MAX_REFERENCE_CODE_FRAMES + 1, N_VQ), dtype=torch.long)
    with pytest.raises(ValueError, match="limit is"):
        decode(too_long)


def test_bad_base64_is_rejected():
    with pytest.raises(ValueError, match="not valid base64"):
        decode({"data": "not base64!!", "shape": [1, N_VQ]})


def test_unsupported_type_is_rejected():
    with pytest.raises(ValueError, match="must be a packed"):
        decode(42)


class RecordingProcessor:
    """Captures what the request builder hands to the processor."""

    model_config = type(
        "FakeModelConfig",
        (),
        {"n_vq": N_VQ, "audio_vocab_size": AUDIO_VOCAB_SIZE},
    )()

    def __init__(self) -> None:
        self.reference = "unset"
        self.conversations = None
        self.mode = None

    def build_user_message(self, **kwargs):
        self.reference = kwargs.get("reference")
        return dict(kwargs, role="user")

    def build_assistant_message(self, audio_codes_list):
        return {"role": "assistant", "audio_codes_list": audio_codes_list}

    def __call__(self, conversations, mode):
        self.conversations = conversations
        self.mode = mode
        rows = torch.full((1, 4, N_VQ + 1), AUDIO_VOCAB_SIZE, dtype=torch.long)
        rows[0, :, 0] = torch.arange(4)
        return {"input_ids": rows}


class ExplodingEncoder:
    """Any call means a codec encode was about to run -- the thing codes remove."""

    def encode(self, source):  # pragma: no cover - must not be reached
        raise AssertionError("reference encoder must not run when ref_codes is given")

    def close(self) -> None:
        pass


def make_payload(tts_params: dict, inputs: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id="req-codes",
        request=OmniRequest(
            inputs=inputs or {"text": "hallo welt"},
            params={},
            metadata={"task": "tts", "tts_params": tts_params},
        ),
        data={},
    )


def run_preprocess(processor, tts_params: dict) -> MossTTSLocalPreparedRequest:
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=ExplodingEncoder()
    )
    try:
        payload = preprocess_moss_tts_local_payload(make_payload(tts_params))
    finally:
        clear_moss_tts_local_preprocessing_context()
    return MossTTSLocalPreparedRequest(**payload.data[PREPARED_MARKER])


def test_ref_codes_resolve_from_tts_params():
    codes = make_codes(3)
    resolved = resolve_moss_ref_codes(
        make_payload({"ref_codes": encode_reference_codes(codes)}),
        processor=RecordingProcessor(),
    )
    assert torch.equal(resolved, codes)


def test_ref_codes_resolve_from_reference_descriptor():
    codes = make_codes(3)
    payload = make_payload(
        {},
        inputs={
            "text": "hallo",
            "references": [{"ref_codes": encode_reference_codes(codes)}],
        },
    )
    assert torch.equal(
        resolve_moss_ref_codes(payload, processor=RecordingProcessor()), codes
    )


def test_absent_ref_codes_resolve_to_none():
    assert (
        resolve_moss_ref_codes(
            make_payload({"ref_audio": "/tmp/x.wav"}), processor=RecordingProcessor()
        )
        is None
    )


def test_prepared_state_stays_msgpack_serializable():
    """The terminal stage hop msgpacks the state dict; a tensor there is fatal."""
    prepared = run_preprocess(
        RecordingProcessor(), {"ref_codes": encode_reference_codes(make_codes(5))}
    )
    wire = prepared.state.to_dict()
    assert not any(isinstance(value, torch.Tensor) for value in wire.values())
    serialize_message(FakeCompleteMessage(wire))


class FakeCompleteMessage:
    """Minimal stand-in for the control-plane message that carries state data."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def to_dict(self) -> dict:
        return {"kind": "complete", "request_id": "req-codes", "data": self.data}


def test_preprocess_routes_codes_and_skips_the_encoder():
    codes = make_codes(9)
    processor = RecordingProcessor()
    prepared = run_preprocess(
        processor,
        {
            # ref_audio present too: codes must win, no encode.
            "ref_audio": "data:audio/wav;base64,AAAA",
            "ref_codes": encode_reference_codes(codes),
        },
    )
    assert processor.mode == "generation"
    assert isinstance(processor.reference, list) and len(processor.reference) == 1
    assert torch.equal(processor.reference[0], codes)
    assert prepared.prefix_codes is None


def test_preprocess_rejects_malformed_codes():
    with pytest.raises(ValueError, match="expected shape"):
        run_preprocess(RecordingProcessor(), {"ref_codes": [[1, 2, 3]]})


def test_continuation_puts_prior_codes_in_the_assistant_slot():
    """Continuation resumes the model's own stream: prior codes are the
    assistant turn, the user turn carries no reference, and the same rows
    travel on as the vocoder's decoder context."""
    codes = make_codes(6)
    processor = RecordingProcessor()
    prepared = run_preprocess(
        processor,
        {"continuation": True, "ref_codes": encode_reference_codes(codes)},
    )
    assert processor.mode == "continuation"
    (conversation,) = processor.conversations
    user, assistant = conversation
    assert user["role"] == "user" and user["reference"] is None
    assert assistant["role"] == "assistant"
    assert torch.equal(assistant["audio_codes_list"][0], codes)
    assert torch.equal(prepared.prefix_codes, codes)


def test_continuation_voice_ref_codes_go_to_the_user_slot():
    prefix = make_codes(6)
    voice = make_codes(3) + 1
    processor = RecordingProcessor()
    run_preprocess(
        processor,
        {
            "continuation": True,
            "ref_codes": encode_reference_codes(prefix),
            "voice_ref_codes": encode_reference_codes(voice),
        },
    )
    user, assistant = processor.conversations[0]
    assert torch.equal(user["reference"][0], voice)
    assert torch.equal(assistant["audio_codes_list"][0], prefix)


# ----------------------------------------------------------------------
#  Admin-plane reference encode (POST /moss/encode_reference)
# ----------------------------------------------------------------------


class FakeReferenceEncoder:
    """Stands in for the preprocessing stage's shared reference encoder."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, source: str) -> torch.Tensor:
        self.seen.append(source)
        if "boom" in source:
            raise ValueError("decode failed")
        return make_codes(3)


def make_handler(encoder):
    return build_reference_encode_admin_handler(encoder, n_vq=N_VQ)


def test_admin_encode_returns_wire_form_codes():
    encoder = FakeReferenceEncoder()
    out = make_handler(encoder)(
        ADMIN_MOSS_ENCODE_REFERENCE, {"audio": "data:audio/flac;base64,AAAA"}
    )
    assert out["success"] is True
    assert out["n_vq"] == N_VQ
    assert out["frames_per_second"] == 12.5
    assert len(out["codes"]) == 1
    # Round-trips through the very validator the generate path applies.
    assert torch.equal(decode(out["codes"][0]), make_codes(3))
    assert encoder.seen == ["data:audio/flac;base64,AAAA"]


def test_admin_encode_preserves_request_order_for_batches():
    encoder = FakeReferenceEncoder()
    items = [f"data:audio/flac;base64,{i}" for i in range(5)]
    out = make_handler(encoder)(ADMIN_MOSS_ENCODE_REFERENCE, {"audio": items})
    assert out["success"] is True
    assert len(out["codes"]) == 5
    assert sorted(encoder.seen) == sorted(items)


def test_admin_encode_result_is_msgpack_serializable():
    """The codes cross a ZMQ control-plane hop; a tensor there would kill it."""
    out = make_handler(FakeReferenceEncoder())(
        ADMIN_MOSS_ENCODE_REFERENCE, {"audio": "data:audio/wav;base64,AAAA"}
    )
    serialize_message(
        AdminResultMessage(
            AdminResult(
                op_id="op",
                stage="preprocessing",
                action=ADMIN_MOSS_ENCODE_REFERENCE,
                success=True,
                data=out,
            )
        )
    )


@pytest.mark.parametrize(
    "bad",
    [None, [], "", ["data:audio/wav;base64,AAAA", 3], "/etc/passwd", "http://x/a.wav"],
)
def test_admin_encode_rejects_anything_but_data_uris(bad):
    """The route is unauthenticated: no local paths, no fetching URLs."""
    encoder = FakeReferenceEncoder()
    out = make_handler(encoder)(ADMIN_MOSS_ENCODE_REFERENCE, {"audio": bad})
    assert out["success"] is False
    assert "audio" in out["error"]
    assert encoder.seen == []


def test_admin_encode_reports_encoder_failure():
    out = make_handler(FakeReferenceEncoder())(
        ADMIN_MOSS_ENCODE_REFERENCE, {"audio": "data:audio/wav;base64,boom"}
    )
    assert out["success"] is False
    assert "decode failed" in out["error"]


def test_admin_handler_opts_out_of_foreign_actions():
    """model_info/weight updates fan out to ALL stages; keep opting out."""
    out = make_handler(FakeReferenceEncoder())("model_info", {})
    assert out["success"] is True
    assert out["unsupported"] is True and out["skipped"] is True


def test_simple_scheduler_has_no_admin_attr_without_handler():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    assert not hasattr(SimpleScheduler(lambda payload: payload), "admin")
    assert hasattr(
        SimpleScheduler(lambda payload: payload, admin_handler=lambda a, p: {}),
        "admin",
    )
