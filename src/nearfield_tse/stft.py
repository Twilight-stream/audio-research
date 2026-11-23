"""Offline and zero-lookahead causal STFT for float32, 16 kHz mono audio."""

import torch
from torch import nn
from torch.nn import functional as F


class STFTFrontend(nn.Module):
    """Matched sqrt-Hann analysis/synthesis with [B, 2, F, T] spectra.

    Require n_fft >= win_length and hop_length to divide win_length with at
    least two overlaps. Squared-window shifts then satisfy COLA (also NOLA).

    Offline mode uses center=True and constant zero padding, including for short
    signals. Causal frame t covers [t*H-(W-H), (t+1)*H), where W=win_length,
    H=hop_length, and the right endpoint is exclusive. It is available after
    (t+1)*H samples, with no future lookahead relative to that boundary.
    Left signal padding is W-H; right signal padding is (-L) % H, used only
    when flushing the final partial hop. T=ceil(L/H). For n_fft > W, each
    already-framed window is zero-extended for the FFT without reading future
    signal samples. This class is a batched frontend, not a streaming state API.

    Causal synthesis uses all overlapping frames; finalizing an output sample
    can require W-H samples of overlap-add delay. Analysis causality does not
    imply zero-delay waveform synthesis after arbitrary spectrum modification.
    """

    def __init__(self, n_fft: int = 512, win_length: int = 512,
                 hop_length: int = 256, causal: bool = False,
                 window_type: str = "sqrt_hann") -> None:
        super().__init__()
        for name, value in (("n_fft", n_fft), ("win_length", win_length),
                            ("hop_length", hop_length)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if n_fft < win_length or win_length % hop_length or 2 * hop_length > win_length:
            raise ValueError("require n_fft >= win_length and hop_length dividing win_length with >=2 overlaps")
        if window_type != "sqrt_hann":
            raise ValueError("window_type must be 'sqrt_hann'")
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.causal = causal
        self.window_type = window_type
        window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        self.register_buffer("window", torch.sqrt(window + 1e-12))

    @staticmethod
    def spec_to_complex(spec: torch.Tensor) -> torch.Tensor:
        """Convert real/imag [B, 2, F, T] to complex [B, F, T]."""
        if spec.ndim != 4 or spec.shape[1] != 2 or spec.dtype not in (torch.float32, torch.float64):
            raise ValueError("spec must be float32/float64 [B, 2, F, T]")
        return torch.complex(spec[:, 0], spec[:, 1])

    @staticmethod
    def complex_to_spec(cspec: torch.Tensor) -> torch.Tensor:
        """Convert complex [B, F, T] to real/imag [B, 2, F, T]."""
        if cspec.ndim != 3 or not cspec.is_complex():
            raise ValueError("cspec must be complex [B, F, T]")
        return torch.stack((cspec.real, cspec.imag), dim=1)

    def forward_stft(self, wav: torch.Tensor) -> torch.Tensor:
        """Analyze nonempty float32 waveform [B, L]; return [B, 2, F, T]."""
        if wav.ndim != 2 or min(wav.shape) < 1 or wav.dtype != torch.float32:
            raise ValueError("wav must be nonempty float32 [B, L]")
        if not self.causal:
            cspec = torch.stft(wav, self.n_fft, self.hop_length, self.win_length,
                               self.window, center=True, pad_mode="constant", return_complex=True)
        else:
            padded = F.pad(wav, (self.win_length - self.hop_length,
                                 (-wav.shape[-1]) % self.hop_length))
            frames = padded.unfold(-1, self.win_length, self.hop_length)
            batch, count, _ = frames.shape
            fft_left = (self.n_fft - self.win_length) // 2
            frames = F.pad(frames, (fft_left, self.n_fft - self.win_length - fft_left))
            # Each independent FFT sees one causal frame and constant zeros only.
            cspec = torch.stft(frames.reshape(batch * count, self.n_fft),
                               self.n_fft, self.hop_length, self.win_length,
                               self.window, center=False, return_complex=True)
            cspec = cspec[..., 0].reshape(batch, count, -1).transpose(1, 2)
        return self.complex_to_spec(cspec)

    def inverse_stft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Synthesize [B, length]; causal mode uses normalized explicit OLA."""
        cspec = self.spec_to_complex(spec)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer")
        expected = ((length + self.hop_length - 1) // self.hop_length if self.causal
                    else 1 + (length + 2 * (self.n_fft // 2) - self.n_fft) // self.hop_length)
        if (spec.dtype != torch.float32 or spec.shape[0] < 1
                or spec.shape[2:] != (self.n_fft // 2 + 1, expected)):
            raise ValueError("spec dtype, frequency bins or frame count do not match frontend/length")
        if not self.causal:
            return torch.istft(cspec, self.n_fft, self.hop_length, self.win_length,
                               self.window, center=True, length=length)
        fft_left = (self.n_fft - self.win_length) // 2
        frames = torch.fft.irfft(cspec, n=self.n_fft, dim=1)
        frames = frames[:, fft_left:fft_left + self.win_length, :] * self.window[None, :, None]
        total = (expected - 1) * self.hop_length + self.win_length
        wav = F.fold(frames, (1, total), (1, self.win_length), stride=(1, self.hop_length))
        weights = self.window.square()[None, :, None].expand(1, -1, expected)
        envelope = F.fold(weights, (1, total), (1, self.win_length), stride=(1, self.hop_length))
        start = self.win_length - self.hop_length
        # Discard padded edges before normalization; retained weights are positive.
        wav = wav[:, 0, 0, start:start + length]
        envelope = envelope[:, 0, 0, start:start + length]
        return wav / envelope.clamp_min(torch.finfo(wav.dtype).tiny)
