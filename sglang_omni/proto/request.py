# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestState(Enum):
    """State of a request in the pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RequestInfo:
    """Tracking info for a request in the coordinator."""

    request_id: str
    state: RequestState = RequestState.PENDING
    current_stage: str | None = None
    terminal_stages: set[str] | None = None
    result: Any = None
    error: str | None = None


EXPLICIT_GENERATION_PARAMS_KEY = "explicit_generation_params"

# MOSS-TTS Local: opt-in echo of the codes the AR engine just generated.
#
# ``metadata.tts_params.return_codes`` (bool) asks the vocoder stage to park the
# generated ``[T, n_vq]`` codes on the terminal payload under
# ``MOSS_GENERATED_CODES_FIELD``, packed in the SAME wire form the ``ref_codes``
# INPUT accepts (base64 int16, C-contiguous row-major, plus shape+dtype). A
# caller that chains segments -- each one conditioned on a window of the
# previous ones -- can then feed the value straight back in instead of paying a
# second codec encode on audio it already had the codes for.
#
# The names live here, next to StagePayload, so the generic serve/client layer
# can read the field without importing torch-backed model code (same reason as
# ADMIN_MOSS_ENCODE_REFERENCE in proto/admin.py). The packing itself is in
# models/moss_tts_local/ref_codes.py, which owns the format.
MOSS_RETURN_CODES_PARAM = "return_codes"
MOSS_GENERATED_CODES_FIELD = "generated_codes"


@dataclass
class OmniRequest:
    """User-facing request with inputs and parameters."""

    inputs: Any
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "_type": "OmniRequest",
            "inputs": self.inputs,
            "params": self.params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmniRequest":
        return cls(
            inputs=data.get("inputs"),
            params=data.get("params", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StagePayload:
    """Payload passed between stages with request context."""

    request_id: str
    request: OmniRequest
    data: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "_type": "StagePayload",
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StagePayload":
        request = data.get("request", {})
        if isinstance(request, dict) and request.get("_type") == "OmniRequest":
            request_obj = OmniRequest.from_dict(request)
        else:
            request_obj = OmniRequest.from_dict(request)
        return cls(
            request_id=data.get("request_id", ""),
            request=request_obj,
            data=data.get("data"),
        )
