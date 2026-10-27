# SPDX-License-Identifier: Apache-2.0
"""Pre-computed reference codes and continuation for MOSS-TTS Delay.

The wire decoder is shared with the Local Transformer and covered in
tests/unit_test/moss_tts_local/test_ref_codes.py. What is specific here is the
Delay path's wiring: Delay runs ``n_vq`` 32 against Local's 12, the
continuation turn keeps the raw prefix grid for the delay seam, and the
repetition penalty groups codebooks the way the reference implementation does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    MossTTSSGLangRequestData,
    build_processor_message,
    prepare_moss_tts_request,
    resolve_moss_ref_codes,
    seed_seam_tail,
)
from sglang_omni.models.moss_tts.sampler import matches_graph_profile
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.ref_codes import decode_reference_codes, encode_reference_codes

N_VQ = 32
N_VQ_LOCAL = 12
AUDIO_VOCAB_SIZE = 1024


def make_codes(frames: int = 7, n_vq: int = N_VQ) -> torch.Tensor:
    return (torch.arange(frames * n_vq, dtype=torch.long) % AUDIO_VOCAB_SIZE).reshape(
        frames, n_vq
    )


class RecordingProcessor:
    """Captures what the request builder hands to the processor."""

    model_config = SimpleNamespace(n_vq=N_VQ, audio_vocab_size=AUDIO_VOCAB_SIZE)

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


class CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, source):
        self.calls += 1
        return make_codes(3)


def make_payload(tts_params: dict, inputs: dict | None = None) -> StagePayload:
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


def test_local_transformer_codes_are_rejected():
    """A 12-channel reference is valid -- for the other model. Accepting it
    here would clone a voice from garbage rather than fail the request."""
    local_codes = encode_reference_codes(make_codes(3, n_vq=N_VQ_LOCAL))
    with pytest.raises(ValueError):
        resolve_moss_ref_codes(
            make_payload({"ref_codes": local_codes}), processor=RecordingProcessor()
        )


def test_codes_win_over_ref_audio_and_skip_the_encoder():
    processor = RecordingProcessor()
    encoder = CountingEncoder()
    codes = make_codes(5)
    state = MossTTSState(text="hallo", ref_audio="/tmp/x.wav", ref_text=None)

    build_processor_message(processor, state, encoder, codes)

    assert encoder.calls == 0
    assert torch.equal(processor.reference[0], codes)


def test_without_codes_the_audio_path_is_untouched():
    processor = RecordingProcessor()
    encoder = CountingEncoder()
    state = MossTTSState(text="hallo", ref_audio="/tmp/x.wav", ref_text=None)

    build_processor_message(processor, state, encoder)

    assert encoder.calls == 1
    assert torch.equal(processor.reference[0], make_codes(3))


def test_continuation_keeps_the_raw_prefix_for_the_seam():
    codes = make_codes(40)
    processor = RecordingProcessor()

    prepared = prepare_moss_tts_request(
        make_payload(
            {"continuation": True, "ref_codes": encode_reference_codes(codes)}
        ),
        processor=processor,
        reference_encoder=CountingEncoder(),
    )

    assert processor.mode == "continuation"
    user, assistant = processor.conversations[0]
    assert user["reference"] is None
    assert torch.equal(assistant["audio_codes_list"][0], codes)
    assert torch.equal(prepared.prefix_codes, codes)


def test_clone_request_has_no_prefix():
    processor = RecordingProcessor()
    prepared = prepare_moss_tts_request(
        make_payload({"ref_codes": encode_reference_codes(make_codes(5))}),
        processor=processor,
        reference_encoder=CountingEncoder(),
    )
    assert processor.mode == "generation"
    assert prepared.prefix_codes is None


def test_seam_tail_is_the_last_n_vq_minus_one_frames():
    codes = make_codes(40)
    data = MossTTSSGLangRequestData(input_ids=torch.zeros(1, dtype=torch.long))
    model = SimpleNamespace(config=SimpleNamespace(n_vq=N_VQ))

    seed_seam_tail(data, codes, model=model)

    assert torch.equal(data.seam_tail, codes[-(N_VQ - 1) :])
    assert data.seam_step == 0


def test_short_prefix_seeds_no_seam_tail():
    data = MossTTSSGLangRequestData(input_ids=torch.zeros(1, dtype=torch.long))
    model = SimpleNamespace(config=SimpleNamespace(n_vq=N_VQ))

    seed_seam_tail(data, make_codes(N_VQ - 2), model=model)

    assert data.seam_tail is None


def test_open_seam_keeps_the_request_off_the_sampling_graph():
    """The captured graph samples every cell; the seam cells must be set."""
    data = MossTTSSGLangRequestData(input_ids=torch.zeros(1, dtype=torch.long))
    assert matches_graph_profile(data)
    data.seam_tail = make_codes(N_VQ - 1)
    assert not matches_graph_profile(data)


def test_repetition_penalty_groups_codebooks_like_the_reference():
    """Codebook 0 against its own history; codebooks 1..n-1 against ONE union."""
    n_vq = 3
    vocab = 8
    history = torch.tensor([[0, 1, 2, 3]])  # text, cb0=1, cb1=2, cb2=3
    data = SimpleNamespace(
        audio_repetition_penalty=2.0,
        prompt_rows=history,
        output_rows=[],
    )
    logits = torch.ones(1, n_vq, vocab)

    MossTTSModelRunner.apply_audio_repetition_penalty(logits, [data], n_vq=n_vq)

    penalised = {
        channel: sorted(torch.nonzero(logits[0, channel] != 1.0).flatten().tolist())
        for channel in range(n_vq)
    }
    assert penalised == {0: [1], 1: [2, 3], 2: [2, 3]}


def test_decoder_is_shared_and_parameterised_by_n_vq():
    for n_vq in (N_VQ_LOCAL, N_VQ):
        codes = make_codes(4, n_vq=n_vq)
        decoded = decode_reference_codes(
            encode_reference_codes(codes), n_vq=n_vq, audio_vocab_size=AUDIO_VOCAB_SIZE
        )
        assert torch.equal(decoded, codes)
