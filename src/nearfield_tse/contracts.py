"""Shared contracts for float32, 16 kHz monaural audio."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AudioMeta:
    """Utterance metadata; distances in meters and times in seconds."""

    utt_id: str
    sample_rate: int = 16000
    scenario: str = "unknown"
    target_distance_m: float | None = None
    interferer_distances_m: tuple[float, ...] = ()
    rt60_s: float | None = None
    t_mix_s: float | None = None
    room_volume_m3: float | None = None
    transcript: str | None = None
    is_target_active: bool = True


@dataclass
class ModelInput:
    """Input: mixture [B, L] float32 at 16 kHz, conventionally in [-1, 1].

    lengths: integer [B] valid sample counts; query_radius: float [B] meters.
    state is an opaque optional streaming container.
    """

    mixture: torch.Tensor
    lengths: torch.Tensor | None = None
    query_radius: torch.Tensor | None = None
    state: Any | None = None


@dataclass
class ModelOutput:
    """Output: est_wav [B, L] float32 at 16 kHz.

    est_spec and mix_spec are float32 [B, 2, F, T], channel 0 real, 1 imag.
    aux holds arbitrary tensor predictions; next_state is opaque streaming state.
    """

    est_wav: torch.Tensor
    est_spec: torch.Tensor
    mix_spec: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    next_state: Any | None = None


def _check_tensor(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D tensor")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or Inf")


def validate_model_io(inp: ModelInput, out: ModelOutput) -> None:
    """Raise ValueError for invalid shapes, dtypes, lengths or nonfinite values.

    Amplitude clipping is deliberately not enforced. Opaque states are untouched.
    """
    _check_tensor("mixture", inp.mixture, 2)
    _check_tensor("est_wav", out.est_wav, 2)
    if out.est_wav.shape != inp.mixture.shape:
        raise ValueError("est_wav.shape must equal mixture.shape [B, L]")
    batch, length = inp.mixture.shape
    if batch == 0 or length == 0:
        raise ValueError("mixture must have nonempty batch and sample dimensions")
    for name, tensor in (("mixture", inp.mixture), ("est_wav", out.est_wav),
                         ("est_spec", out.est_spec), ("mix_spec", out.mix_spec)):
        if name.endswith("spec"):
            _check_tensor(name, tensor, 4)
            if tensor.shape[:2] != (batch, 2) or min(tensor.shape[2:]) < 1:
                raise ValueError(f"{name} must have shape [B, 2, F, T], F,T > 0")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
    if out.est_spec.shape != out.mix_spec.shape:
        raise ValueError("est_spec and mix_spec shapes must match")
    for name, tensor in (("lengths", inp.lengths), ("query_radius", inp.query_radius)):
        if tensor is not None:
            _check_tensor(name, tensor, 1)
            if tensor.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
    if inp.lengths is not None:
        if inp.lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be int32 or int64")
        if ((inp.lengths < 0) | (inp.lengths > length)).any().item():
            raise ValueError("lengths must lie in [0, L]")
    for name, tensor in out.aux.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"aux[{name!r}] must be a tensor")
        _check_tensor(f"aux[{name!r}]", tensor, tensor.ndim)
