# Step 1 environment audit

Audited: 2026-11-23, approximately 14:35-14:36 Asia/Shanghai (UTC+08:00).
Workspace: `D:\Audio Research`. Shell: PowerShell. No packages installed or upgraded.

## OS and compute

| Probe | Observed value |
| --- | --- |
| Python platform string | `Windows-10-10.0.26200-SP0` |
| Windows registry ProductName | `Windows 10 Home` |
| Registry DisplayVersion / EditionID | `25H2` / `Core` |
| Registry build / UBR | `26200` / `9974` |
| Architecture | 64-bit AMD64 (Python MSC v.1916) |
| CPU registry model | `AMD Ryzen 7 7840H w/ Radeon 780M Graphics` (spacing normalized) |
| Logical CPU count (`os.cpu_count`) | 16 |
| Total physical RAM (`GlobalMemoryStatusEx`) | 16,415,322,112 bytes = 15.287960 GiB |
| NVIDIA GPU count (`nvidia-smi`) | 1 |
| GPU model | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM per GPU | 8,188 MiB |
| NVIDIA driver | 592.82 |

These OS strings are reported verbatim; the registry product label is not used
to infer a different Windows marketing name. CIM queries were denied in the
sandbox; registry reads and the Win32 memory API supplied the facts above.

## Workspace disk

`df` is unavailable in this Windows environment. Native equivalent:
`shutil.disk_usage('.')`, on the workspace's D: volume.

| Capacity | Bytes | GiB |
| --- | ---: | ---: |
| Total | 257,878,753,280 | 240.168304 |
| Used | 90,638,057,472 | 84.413269 |
| Free | 167,240,695,808 | 155.755035 |

## Python and PyTorch

- Interpreter: `D:\Anaconda\python.exe`.
- Python: `3.11.7 | packaged by Anaconda, Inc. | (main, Dec 15 2023, 18:05:47) [MSC v.1916 64 bit (AMD64)]`.
- PyTorch imported successfully: `torch.__version__ == '2.4.0+cpu'`.
- PyTorch distribution metadata: `2.4.0`.
- `torch.cuda.is_available()`: `False`.
- `torch.version.cuda`: `None` (CPU-only build).
- `torch.cuda.device_count()`: `0`.
- A physical NVIDIA GPU is present, but this interpreter cannot use CUDA through PyTorch.
- `setuptools`: `68.2.2`; `pip`: `26.2.2`.

## Requested package versions

Versions below come from `importlib.metadata.version`; only PyTorch's runtime
import/CUDA status was probed. Metadata presence does not prove extension or ABI
compatibility. `NOT_INSTALLED` means no distribution metadata in this interpreter.

| Package | Installed version |
| --- | --- |
| torchaudio | 2.4.0 |
| soundfile | 0.14.0 |
| scipy | 1.10.1 |
| numpy | 1.26.4 |
| einops | NOT_INSTALLED |
| pyyaml | 6.0.1 |
| pytest | 7.4.0 |
| pyroomacoustics | 0.10.1 |
| gpuRIR | NOT_INSTALLED |
| rotary_embedding_torch | NOT_INSTALLED |
| onnx | NOT_INSTALLED |
| onnxruntime | NOT_INSTALLED |
| sherpa_onnx | NOT_INSTALLED |
| jiwer | NOT_INSTALLED |
| pesq | NOT_INSTALLED |
| pystoi | NOT_INSTALLED |
| mamba_ssm | NOT_INSTALLED |
| causal_conv1d | NOT_INSTALLED |
| flash_attn | NOT_INSTALLED |
| transformers | 5.4.0 |

## Network and package access

- Sandboxed Python HTTPS probes to `https://example.com/` and
  `https://pypi.org/simple/pytest/` failed with `PermissionError`, Winsock 10013.
- The same read-only probes outside the sandbox both returned HTTP 200 at
  `2026-11-23T06:36:25.043170+00:00` (14:36:25 Asia/Shanghai).
- Outbound HTTPS and the public PyPI simple index are reachable outside the
  sandbox; sandboxed shell execution is network-restricted, not evidence that
  the host is offline-only. Package downloads/installations were not attempted.
- Existing torch, numpy, PyYAML and pytest suffice for Step 1. CUDA execution
  cannot be validated with the active CPU-only PyTorch build.

## Verification scope

Tests use the existing interpreter and CPU, with the source tree on pytest's
Python path through `pyproject.toml`; no editable install is required. Coverage
includes the requested long-signal round trips, short/nonaligned signals,
zero-lookahead causal perturbation and impulse tests, explicit overlap-add,
FFT zero padding, gradients, contracts, registries, YAML inheritance/overrides
and RNG seeding. No neural model, data simulation, training, loss or ASR wrapper
is implemented in this step.
