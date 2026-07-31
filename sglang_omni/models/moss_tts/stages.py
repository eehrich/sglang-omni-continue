# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.moss_tts.codec import (
    split_moss_audio_segments,
    split_moss_audio_segments_with_prefix,
)
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)
from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    moss_tts_special_token_defaults,
)
from sglang_omni.models.moss_tts.request_builders import (
    _reference_for_processor,
    cleanup_prepared_moss_tts_request,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.proto import (
    MOSS_GENERATED_CODES_FIELD,
    MOSS_RETURN_CODES_PARAM,
    StagePayload,
)
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.ref_codes import (
    ADMIN_ENCODE_REFERENCE,
    CODEC_FRAMES_PER_SECOND,
    encode_reference_codes,
)

logger = logging.getLogger(__name__)


def _drop_prefix_samples(
    waveform: torch.Tensor, prefix_frames: int, segment_frames: int
) -> torch.Tensor:
    """Cut the samples the continuation prefix produced off the front.

    The samples-per-frame ratio comes from THIS decode rather than from a
    configured rate: the codec emits whole frames, so the division is exact,
    and deriving it here cannot drift if the rate ever changes. A remainder
    means that assumption broke -- fall back to a proportional cut and say so,
    because the broken assumption is the interesting part, not the half sample.
    """
    if prefix_frames <= 0 or segment_frames <= 0:
        return waveform
    samples = int(waveform.shape[-1])
    per_frame, remainder = divmod(samples, segment_frames)
    if remainder:
        logger.warning(
            "MOSS-TTS vocoder: %d samples over %d frames is not a whole ratio "
            "-- cutting the prefix proportionally",
            samples,
            segment_frames,
        )
        offset = int(round(prefix_frames * samples / segment_frames))
    else:
        offset = prefix_frames * per_frame
    if offset >= samples:
        logger.error(
            "MOSS-TTS vocoder: prefix of %d frames covers the whole segment "
            "(%d samples) -- not cutting",
            prefix_frames,
            samples,
        )
        return waveform
    return waveform[..., offset:].contiguous()

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)


def load_state(payload: StagePayload) -> MossTTSState:
    return _load_pipeline_state(payload, MossTTSState)


def store_state(payload: StagePayload, state: MossTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _load_moss_processor(
    model_path: str,
    *,
    device: str = "cpu",
    dtype: str | torch.dtype = "float32",
) -> Any:
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    logger.info(f"Loading MOSS-TTS processor from {checkpoint_dir} on {device}")
    try:
        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(checkpoint_dir)
            processor = processor_cls.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    _normalize_moss_processor_config(processor)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None:
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()
        if hasattr(audio_tokenizer, "to"):
            kwargs: dict[str, Any] = {"device": device}
            if device != "cpu":
                kwargs["dtype"] = _torch_dtype(dtype)
            audio_tokenizer.to(**kwargs)
    return processor


def build_reference_encode_admin_handler(processor: Any, *, n_vq: int) -> Any:
    """Admin-plane handler that turns audio into ``ref_codes`` wire payloads.

    Same contract as the Local Transformer's: POST /moss/encode_reference
    dispatches here, and a caller encodes a voice once instead of re-shipping
    and re-encoding it on every request. Without it that route answers 501 on
    a Delay pipeline, and a client that wants codes has to load the processor
    itself just to compute them.

    The encode runs through the SAME path a request would take
    (``_reference_for_processor``), so the codes are what an in-request
    ``ref_audio`` would have produced.

    Unknown actions must report themselves as unsupported rather than fail:
    ``/model_info`` and the weight-update actions fan out to ALL stages, and
    this stage has to keep opting out of them without breaking the aggregate.
    """

    def _encode_one(item: str) -> dict[str, Any]:
        reference = _reference_for_processor(processor, item)
        if not reference:
            raise ValueError("reference audio produced no codes")
        return encode_reference_codes(
            torch.as_tensor(reference[0], dtype=torch.long)[:, :n_vq].contiguous()
        )

    def handler(action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action != ADMIN_ENCODE_REFERENCE:
            return {
                "success": True,
                "message": "stage does not support admin operations",
                "skipped": True,
                "unsupported": True,
            }
        raw = payload.get("audio")
        items = [raw] if isinstance(raw, str) else list(raw or [])
        if not items or not all(isinstance(x, str) and x for x in items):
            return {
                "success": False,
                "error": (
                    f"{ADMIN_ENCODE_REFERENCE}: 'audio' must be a non-empty "
                    "data URI string or a list of them"
                ),
            }
        try:
            codes = [_encode_one(item) for item in items]
        except Exception as exc:  # noqa: BLE001 - report, do not kill the stage
            return {"success": False, "error": f"reference encode failed: {exc}"}
        return {
            "success": True,
            "codes": codes,
            "n_vq": int(n_vq),
            "frames_per_second": float(CODEC_FRAMES_PER_SECOND),
        }

    return handler


def create_preprocessing_executor(
    model_path: str, *, max_concurrency: int = 8
) -> SimpleScheduler:
    processor = _load_moss_processor(model_path, device="cpu", dtype="float32")
    set_moss_tts_preprocessing_context(processor=processor)
    # Preprocessing is CPU-heavy: every request tokenizes text and encodes the
    # reference audio through the MOSS audio tokenizer. Serial execution
    # (max_concurrency=1) lets the codec encode dominate wall-clock and starves
    # the AR engine to batch size 1 (the dominant RTF cost). Run several in
    # parallel — threads release the GIL during the torch codec forward — so the
    # AR OmniScheduler receives a steady, batchable request stream. Mirrors the
    # fishaudio_s2_pro preprocessing stage, which encodes references the same way.
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
        max_concurrency=max_concurrency,
        admin_handler=build_reference_encode_admin_handler(
            processor, n_vq=int(processor.model_config.n_vq)
        ),
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder

    return MossTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


class _MossTTSVocoder(BatchVocoderBase):
    def __init__(self, processor: Any, device: str) -> None:
        self._processor = processor
        self._device = device

    def prepare_item(self, payload: StagePayload) -> tuple[MossTTSState, torch.Tensor]:
        state = load_state(payload)
        if state.delayed_audio_codes is None:
            raise RuntimeError("MOSS-TTS vocoder requires delayed_audio_codes")
        delayed_codes = torch.as_tensor(state.delayed_audio_codes, dtype=torch.long)
        if delayed_codes.numel() == 0:
            raise RuntimeError("MOSS-TTS generated no delayed audio codes")
        return state, delayed_codes

    def _decode_audio(
        self,
        state: MossTTSState,
        delayed_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        delayed_codes = delayed_codes.to(device=self._device, dtype=torch.long)
        audio_pad_code = int(
            getattr(
                getattr(self._processor, "model_config", None),
                "audio_pad_code",
                1024,
            )
        )
        # The prefix STAYS in the codes here and is cut off the audio below:
        # the codec carries state across frames, so decoding a continuation's
        # new frames alone starts it cold. Measured on the 1.7B path, where
        # the same defect sat one layer later, that is about ten semitones too
        # high at the segment start, settling over some ten seconds. The echo
        # (_pack_generated_codes) keeps cutting the CODES -- a caller chains
        # them into the next request and must not get the prefix back.
        segments, prefix_frames = split_moss_audio_segments_with_prefix(
            delayed_codes,
            audio_pad_code=audio_pad_code,
            assistant_start_length=int(state.assistant_start_length),
        )
        decoded = []
        for segment in segments:
            decoded.extend(self._processor.decode_audio_codes([segment]))
        if not decoded:
            raise RuntimeError("MOSS-TTS vocoder decoded no audio segments")
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        waveforms[0] = _drop_prefix_samples(
            waveforms[0], prefix_frames, int(segments[0].shape[0])
        )
        waveform = torch.cat(waveforms, dim=0)
        sample_rate = int(
            getattr(getattr(self._processor, "model_config", None), "sampling_rate", 0)
            or getattr(
                getattr(
                    getattr(self._processor, "audio_tokenizer", None), "config", None
                ),
                "sampling_rate",
                0,
            )
            or state.sample_rate
            or 24000
        )
        return waveform, sample_rate

    async def decode_batch(
        self, items: list[tuple[MossTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        return [self._decode_audio(state, codes) for state, codes in items]

    @staticmethod
    def _return_codes_requested(payload: StagePayload) -> bool:
        """Did this request opt in to getting its generated codes back?

        Opt-in rather than always-on: the packed codes are ~2 KB per generated
        second, which every existing caller would pay for on a response it
        never reads.
        """
        metadata = getattr(getattr(payload, "request", None), "metadata", None)
        if not isinstance(metadata, dict):
            return False
        tts_params = metadata.get("tts_params")
        if not isinstance(tts_params, dict):
            return False
        return bool(tts_params.get(MOSS_RETURN_CODES_PARAM))

    def _pack_generated_codes(self, state: MossTTSState) -> dict[str, Any] | None:
        """Echo the just-generated codes back in the ``ref_codes`` wire form.

        A caller chaining segments -- a re-anchored window, where segment N is
        conditioned on N-1 -- would otherwise have to ship the vocoded WAV back
        for a second codec encode. On the Delay model that encode is the one
        that materialises O(T^2) attention state for a long window, so handing
        the codes back is what makes the chain affordable here at all.

        Undelayed on purpose: ``delayed_audio_codes`` carries the delay pattern
        the AR engine emits, while ``ref_codes`` expects the plain [T, n_vq]
        rows a reference has. ``split_moss_audio_segments`` is the same reversal
        the vocoder itself runs, so the echo describes exactly the audio the
        caller receives.

        Packed rather than raw: the terminal payload is msgpack'd on the stage
        hop, and the packed dict is byte-identical to what ``ref_codes`` takes.

        Returns the packed dict instead of writing it straight onto the
        payload, because ``store_state`` REPLACES ``payload.data`` further
        down -- anything attached before that call is silently dropped.
        """
        try:
            delayed = state.delayed_audio_codes
            if delayed is None:
                return None
            cfg = getattr(self._processor, "model_config", None)
            segments = split_moss_audio_segments(
                torch.as_tensor(delayed, dtype=torch.long),
                audio_pad_code=int(getattr(cfg, "audio_pad_code", 1024)),
                assistant_start_length=int(state.assistant_start_length),
            )
            if not segments:
                return None
            rows = torch.cat(segments, dim=0).detach().to("cpu", torch.long)
            n_vq = int(getattr(cfg, "n_vq", rows.shape[1]))
            return encode_reference_codes(rows[:, :n_vq].contiguous())
        except Exception:
            # Never fail a finished generation over an optional echo -- the
            # caller falls back to encoding the returned WAV.
            logger.warning("MOSS-TTS: could not pack generated codes", exc_info=True)
            return None

    def store_result(
        self,
        payload: StagePayload,
        state: MossTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_payload = audio_waveform_payload(wav, source_hint="MOSS-TTS")
        # Read before the clear below: this is the last point the generated
        # codes exist on the request.
        generated_codes = (
            self._pack_generated_codes(state)
            if self._return_codes_requested(payload)
            else None
        )
        state.delayed_audio_codes = None
        state.sample_rate = int(sample_rate)
        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        # After store_state, which replaces payload.data wholesale.
        if generated_codes is not None:
            payload.data[MOSS_GENERATED_CODES_FIELD] = generated_codes
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    processor = _load_moss_processor(model_path, device=device, dtype=dtype)

    return _MossTTSVocoder(processor, device).build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
