"""Step 1 contracts, utilities, numerical reconstruction and causality checks."""

from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from nearfield_tse.contracts import AudioMeta, ModelInput, ModelOutput, validate_model_io
from nearfield_tse.registry import LOSS_REGISTRY, MODEL_REGISTRY, POSTPROC_REGISTRY, Registry
from nearfield_tse.stft import STFTFrontend
from nearfield_tse.utils.config import load_config
from nearfield_tse.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("causal,n_fft,win,hop", [(False, 512, 512, 256), (True, 400, 400, 100)])
@pytest.mark.parametrize("length", [16000, 16001, 32123])
def test_required_reconstruction(causal: bool, n_fft: int, win: int, hop: int, length: int) -> None:
    frontend = STFTFrontend(n_fft, win, hop, causal)
    wav = torch.rand(2, length, generator=torch.Generator().manual_seed(length)) * 2 - 1
    spec = frontend.forward_stft(wav)
    count = (length + hop - 1) // hop if causal else length // hop + 1
    assert spec.shape == (2, 2, n_fft // 2 + 1, count)
    assert spec.dtype == torch.float32
    reconstructed = frontend.inverse_stft(spec, length)
    assert reconstructed.shape == wav.shape
    assert (reconstructed - wav).abs().max().item() < 1e-4


@pytest.mark.parametrize("causal,n_fft,win,hop", [
    (False, 512, 512, 256), (True, 512, 512, 256),
    (True, 400, 400, 100), (True, 512, 400, 100),
    (True, 513, 400, 100), (False, 513, 400, 100),
])
@pytest.mark.parametrize("length", [1, 99, 100, 101, 399, 400, 401, 513])
def test_short_and_fft_padded_reconstruction(causal: bool, n_fft: int, win: int,
                                           hop: int, length: int) -> None:
    frontend = STFTFrontend(n_fft, win, hop, causal)
    wav = torch.randn(1, length, generator=torch.Generator().manual_seed(length))
    reconstructed = frontend.inverse_stft(frontend.forward_stft(wav), length)
    assert reconstructed.shape == wav.shape
    assert (reconstructed - wav).abs().max().item() < 1e-4


@pytest.mark.parametrize("n_fft", [400, 512, 513])
@pytest.mark.parametrize("split", [0, 1, 99, 100, 101, 1234, 7999, 16000])
def test_strict_causality_perturbation(n_fft: int, split: int) -> None:
    frontend = STFTFrontend(n_fft, 400, 100, causal=True)
    generator = torch.Generator().manual_seed(42)
    x1 = torch.randn(2, 16001, generator=generator)
    x2 = x1.clone()
    x2[:, split:] = torch.randn(2, x1.shape[-1] - split, generator=generator)
    s1, s2 = frontend.forward_stft(x1), frontend.forward_stft(x2)
    ends = (torch.arange(s1.shape[-1]) + 1) * frontend.hop_length
    past = ends <= split
    assert torch.equal(s1[..., past], s2[..., past])
    assert (s1[..., ~past] - s2[..., ~past]).abs().max().item() > 0


@pytest.mark.parametrize("n_fft", [400, 512, 513])
def test_causal_impulse_alignment_and_tail(n_fft: int) -> None:
    frontend = STFTFrontend(n_fft, 400, 100, causal=True)
    wav = torch.zeros(1, 1001)
    wav[0, 100] = 1
    wav[0, -1] = -1
    spec = frontend.forward_stft(wav)
    assert torch.count_nonzero(spec[..., 0]).item() == 0  # frame 0 ends at 100
    assert torch.count_nonzero(spec[..., 1]).item() > 0
    assert torch.count_nonzero(spec[..., -1]).item() > 0
    padded = F.pad(wav, (300, 99))
    frames = padded.unfold(-1, 400, 100) * frontend.window
    left = (n_fft - 400) // 2
    expected = torch.fft.rfft(F.pad(frames, (left, n_fft - 400 - left)), n=n_fft).transpose(1, 2)
    torch.testing.assert_close(frontend.spec_to_complex(spec), expected, rtol=1e-5, atol=1e-6)
    assert (frontend.inverse_stft(spec, 1001) - wav).abs().max().item() < 1e-4


@pytest.mark.parametrize("causal", [False, True])
def test_window_buffer_cola_silence_and_gradient(causal: bool) -> None:
    frontend = STFTFrontend(400, 400, 100, causal)
    assert dict(frontend.named_buffers())["window"] is frontend.window
    assert "window" in frontend.state_dict()
    assert not list(frontend.parameters())
    cola = sum(frontend.window.square().roll(shift) for shift in range(0, 400, 100))
    torch.testing.assert_close(cola, torch.full((400,), 2.0))
    silence = torch.zeros(2, 1001)
    assert torch.equal(frontend.inverse_stft(frontend.forward_stft(silence), 1001), silence)
    wav = torch.randn(1, 1001, requires_grad=True)
    frontend.inverse_stft(frontend.forward_stft(wav), 1001).square().mean().backward()
    assert wav.grad is not None and torch.isfinite(wav.grad).all()
    torch.testing.assert_close(wav.grad, 2 * wav.detach() / wav.numel(), rtol=1e-3, atol=1e-5)


def test_complex_conversions_noncontiguous() -> None:
    cspec = torch.randn(2, 5, 7, dtype=torch.complex64).transpose(1, 2)
    spec = STFTFrontend.complex_to_spec(cspec)
    assert spec.shape == (2, 2, 7, 5)
    assert torch.equal(spec[:, 0], cspec.real)
    assert torch.equal(spec[:, 1], cspec.imag)
    assert torch.equal(STFTFrontend.spec_to_complex(spec), cspec)
    with pytest.raises(ValueError):
        STFTFrontend.spec_to_complex(torch.zeros(2, 3, 5, 7))
    with pytest.raises(ValueError):
        STFTFrontend.complex_to_spec(torch.zeros(2, 5, 7))


@pytest.mark.parametrize("kwargs", [
    {"n_fft": 0}, {"n_fft": True}, {"n_fft": 1.5}, {"win_length": 0},
    {"hop_length": 0}, {"n_fft": 200}, {"hop_length": 200},
    {"hop_length": 512}, {"window_type": "unknown"},
])
def test_frontend_bad_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        STFTFrontend(**kwargs)


@pytest.mark.parametrize("causal", [False, True])
def test_frontend_bad_io(causal: bool) -> None:
    frontend = STFTFrontend(causal=causal)
    for wav in [torch.zeros(16), torch.zeros(1, 1, 16), torch.zeros(1, 0),
                torch.zeros(0, 16), torch.zeros(1, 16, dtype=torch.float64)]:
        with pytest.raises(ValueError):
            frontend.forward_stft(wav)
    spec = frontend.forward_stft(torch.zeros(1, 16001))
    for bad in [spec[:, :, :-1], spec[..., :-1], spec.double(), spec[:0]]:
        with pytest.raises(ValueError):
            frontend.inverse_stft(bad, 16001)
    for length in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            frontend.inverse_stft(spec, length)


def _valid_io() -> tuple[ModelInput, ModelOutput]:
    inp = ModelInput(torch.zeros(2, 1000), lengths=torch.tensor([1000, 900]),
                     query_radius=torch.tensor([1.0, 2.0]), state=object())
    out = ModelOutput(torch.zeros(2, 1000), torch.zeros(2, 2, 201, 10),
                      torch.zeros(2, 2, 201, 10), {"target_active_logits": torch.zeros(2)}, object())
    return inp, out


def test_contracts_valid_and_defaults() -> None:
    inp, out = _valid_io()
    validate_model_io(inp, out)
    validate_model_io(ModelInput(inp.mixture), replace(out, aux={}))
    assert AudioMeta("utt").sample_rate == 16000
    assert AudioMeta("utt").interferer_distances_m == ()
    assert AudioMeta("utt").is_target_active
    fresh = ModelOutput(out.est_wav, out.est_spec, out.mix_spec)
    other = ModelOutput(out.est_wav, out.est_spec, out.mix_spec)
    fresh.aux["x"] = torch.zeros(1)
    assert not other.aux


@pytest.mark.parametrize("field,shape", [
    ("mixture", (2, 1, 1000)), ("mixture", (0, 1000)), ("mixture", (2, 0)),
    ("est_wav", (2, 999)), ("est_wav", (1, 1000)), ("est_wav", (2, 1, 1000)),
    ("est_spec", (2, 201, 10)), ("est_spec", (1, 2, 201, 10)),
    ("est_spec", (2, 3, 201, 10)), ("est_spec", (2, 2, 200, 10)),
    ("mix_spec", (2, 2, 201, 9)), ("mix_spec", (2, 2, 201, 0)),
    ("lengths", (2, 1)), ("query_radius", (1,)),
])
def test_contract_shape_errors(field: str, shape: tuple[int, ...]) -> None:
    inp, out = _valid_io()
    if field in {"mixture", "lengths", "query_radius"}:
        inp = replace(inp, **{field: torch.zeros(shape)})
    else:
        out = replace(out, **{field: torch.zeros(shape)})
    with pytest.raises(ValueError):
        validate_model_io(inp, out)


@pytest.mark.parametrize("field", ["mixture", "est_wav", "est_spec", "mix_spec", "query_radius", "aux"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_contract_nonfinite_errors(field: str, value: float) -> None:
    inp, out = _valid_io()
    if field == "aux":
        out.aux["bad"] = torch.tensor(value)
    else:
        getattr(inp if field in {"mixture", "query_radius"} else out, field).flatten()[0] = value
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_model_io(inp, out)


def test_contract_dtype_and_length_errors() -> None:
    inp, out = _valid_io()
    for lengths in [torch.tensor([-1, 1]), torch.tensor([1, 1001]), torch.ones(2),
                    torch.tensor([float("nan"), 1])]:
        with pytest.raises(ValueError):
            validate_model_io(replace(inp, lengths=lengths), out)
    for field in ["mixture", "est_wav", "est_spec", "mix_spec"]:
        with pytest.raises(ValueError, match="float32"):
            if field == "mixture":
                validate_model_io(replace(inp, mixture=inp.mixture.double()), out)
            else:
                validate_model_io(inp, replace(out, **{field: getattr(out, field).double()}))
    with pytest.raises(ValueError, match="tensor"):
        validate_model_io(inp, replace(out, aux={"bad": None}))


def test_registry() -> None:
    registry: Registry[object] = Registry("test")

    @registry.register("example")
    class Example:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    assert registry.get("example") is Example
    assert registry.build("example", value=5).value == 5
    assert registry.list() == ["example"]
    listed = registry.list()
    listed.clear()
    assert registry.list() == ["example"]
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("example")(Example)
    for action in [registry.get, registry.build]:
        with pytest.raises(KeyError, match="unknown"):
            action("missing")
    with pytest.raises(ValueError):
        registry.register(" ")
    with pytest.raises(ValueError, match="callable"):
        registry.register("bad")(None)
    assert MODEL_REGISTRY.list() == LOSS_REGISTRY.list() == POSTPROC_REGISTRY.list() == []
    assert len({id(MODEL_REGISTRY), id(LOSS_REGISTRY), id(POSTPROC_REGISTRY)}) == 3


def test_base_config_and_typed_overrides() -> None:
    config = load_config(ROOT / "configs/base.yaml")
    assert config == {"sample_rate": 16000, "stft": {"n_fft": 512, "win_length": 512,
                    "hop_length": 256, "causal": False}, "seed": 42}
    overridden = load_config(ROOT / "configs/base.yaml", ["stft.n_fft=400", "training.lr=1e-4",
        "stft.causal=true", "training.items=[1, 2]", "training.none=null", 'training.label="1e-4"'])
    assert overridden["stft"]["n_fft"] == 400
    assert overridden["stft"]["causal"] is True
    assert overridden["training"] == {"lr": 1e-4, "items": [1, 2], "none": None, "label": "1e-4"}
    assert config["stft"]["n_fft"] == 512


def test_recursive_relative_config_inheritance(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "root.yaml").write_text("stft: {n_fft: 512, hop_length: 256}\nseed: 42\n", encoding="utf-8")
    (tmp_path / "nested/first.yaml").write_text("base_config: ../root.yaml\nstft: {n_fft: 400}\n", encoding="utf-8")
    (tmp_path / "second.yaml").write_text("stft: {hop_length: 100}\nseed: 7\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("base_config: [nested/first.yaml, second.yaml]\nstft: {causal: true}\n", encoding="utf-8")
    result = load_config(child, ["seed=9"])
    assert result == {"stft": {"n_fft": 400, "hop_length": 100, "causal": True}, "seed": 9}


def test_config_cycle_and_invalid_documents(tmp_path: Path) -> None:
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text("base_config: b.yaml\n", encoding="utf-8")
    second.write_text("base_config: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(first)
    for contents in ["[]", "42", "1: value", "base_config: 42", "base_config: [42]"]:
        first.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(first)
    first.write_text("", encoding="utf-8")
    assert load_config(first) == {}
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("override", ["missing_equals", "=1", ".key=1", "a..b=1", "a.=1", "a. b=1", "seed.value=1"])
def test_invalid_config_overrides(override: str) -> None:
    with pytest.raises(ValueError):
        load_config(ROOT / "configs/base.yaml", [override])


def test_seeding_reproducibility_and_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        seed_everything(42, deterministic=True)
        first = (random.random(), np.random.rand(4), torch.rand(4))
        assert torch.are_deterministic_algorithms_enabled()
        seed_everything(42, deterministic=True)
        second = (random.random(), np.random.rand(4), torch.rand(4))
        assert first[0] == second[0]
        np.testing.assert_array_equal(first[1], second[1])
        assert torch.equal(first[2], second[2])
    finally:
        seed_everything(42, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    for seed in [-1, 2**32, True, 1.5]:
        with pytest.raises(ValueError):
            seed_everything(seed)
