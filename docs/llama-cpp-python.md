# llama-cpp-python: versions, devices, and how the GGUF backend gets built

The GGUF translation models (Gemma Q6/Q8, VNTL Llama-3) run through
`llama_cpp`, a ctypes wrapper around a shared library that does not exist
until someone compiles it. This document covers which version we compile,
for which devices, with which flags — and why each choice is the way it is.

The implementation lives in `packaging/llamacpp.py`. The build driver that
calls it is `packaging/build.py` (`install_llama_cpp`, step 5/15).

---

## 1. Version: pinned, locked, compiled from source

| Item | Value | Where |
|---|---|---|
| Package | `llama-cpp-python` | `pyproject.toml`, `gguf` extra |
| Pinned version | `==0.3.35` | `uv.lock` |
| Vendored llama.cpp | ggml `0.20.0` (commit `4df29be`) | inside the sdist, `vendor/llama.cpp` |
| Torch for cu118 | `2.7.1+cu118` (last release on the cu118 index) | `uv.lock` |
| Torch for cu129_win | `2.8.0+cu129` | `uv.lock` |
| Torch for everything else | `2.13.0` (+cpu/cu126/cu129/cu130/cu132/rocm7.1/rocm7.2) | `uv.lock` |

Three rules follow from the fact that PyPI ships no wheels for this
package — only an sdist:

1. **Every install compiles llama.cpp.** What the binary can do is decided
   entirely by `CMAKE_ARGS`. There is no "CUDA wheel" to download.
2. **The version is the locked one.** `locked_version()` reads it out of
   `uv.lock`; an install with no pinned version is refused rather than
   resolved fresh, so two builds of one commit never compile different
   llama.cpp releases.
3. **No cache.** Installs use `uv pip install --no-cache-dir`, because uv's
   wheel cache is keyed on the sdist, not on `CMAKE_ARGS` — without the flag
   a CUDA build silently reuses yesterday's CPU wheel and segfaults on the
   first model load.

---

## 2. Devices: one name, two decisions

`--device` picks a **torch build** (via a `[tool.uv.sources]` index) *and* a
**llama.cpp backend** (via generated `CMAKE_ARGS`). Getting the second one
wrong is quiet — a CUDA build with a CPU-only `libllama` loads, answers, and
translates at a tenth of the speed with nothing in the log — so an argument
list that cannot be satisfied is a build failure, never a downgrade.

| `--device` | Torch index | Torch | Needs toolkit | CI runner |
|---|---|---|---|---|
| `cpu` | CPU | 2.13.0 | none | ubuntu-24.04 / windows-latest |
| `macos` | PyPI | 2.13.0 | none (Metal) | Mac only |
| `cu118` | cu118 | **2.7.1** | CUDA 11.8 + gcc-11 | ubuntu-22.04 / windows-latest |
| `cu126` | cu126 | 2.13.0 | CUDA 12.6 + gcc-13 | ubuntu-24.04 / windows-latest |
| `cu129` | cu129 | 2.13.0 | CUDA 12.9 + gcc-13 | ubuntu-24.04 |
| `cu129_win` | cu129 (Windows split) | 2.8.0 | CUDA 12.9 + MSVC | windows-latest |
| `cu130` | cu130 | 2.13.0 | CUDA 13.0 + gcc-14 | ubuntu-24.04 / windows-latest |
| `cu132` | cu132 | 2.13.0 | CUDA 13.2 + gcc-14 | — (no CI row yet; see notes below) |
| `rocm71` / `rocm72` | rocm7.1 / rocm7.2 | 2.13.0 | ROCm 6.1+ | Linux only |

Notes:

* `cu118` is stuck on torch 2.7.1 because that is the newest release the
  cu118 index carries; `cu129_win` tracks 2.8.0. Everything else tracks 2.13.0.
* `cu129_win` exists because the Windows CUDA wheel lives on a different
  PyPI index split than the Linux one; same toolkit either way.
* **13.x torch landscape (PyTorch 2.11+).** Plain `pip install torch` now
  ships CUDA **13.0** wheels by default; `cu126` stays as the fallback for
  older drivers and pre-Turing GPUs; `cu128` is deprecated in favour of
  `cu130`. `cu132` is wired through `pyproject.toml` (extra, index, sources)
  so `python packaging/build.py --device cu132` works locally, but
  `.github/workflows/build.yml` has no matrix row for it yet — add
  `{ name: cu132-gguf-linux, os: ubuntu-24.04, device: cu132,
  cuda: "13.2.x", gcc: "14", extra: "" }` (plus `-windows` / `--no-gguf`
  twins) to enable it in CI. It compiles the identical arch list to 13.0
  (13.2 adds no new architectures, only toolchain fixes).
* ROCm wheels exist for Linux only, so `--device rocm*` is refused on
  Windows — but `--gguf-device rocm72` with `--device cpu` works there for
  the translator backend alone.
* The name is pattern-based, not a lookup table: a future `cu133` extra in
  `pyproject.toml` works with no code change; the CUDA version is read out
  of the digits.

### Related build flags

| Flag | Effect |
|---|---|
| `--no-gguf` | Skip the backend entirely (no OpenBLAS/CUDA needed; GGUF models won't load) |
| `--gguf-device DEVICE` | Build llama.cpp for another device than `--device`, e.g. CUDA torch + CPU translator for a machine whose toolkit can't compile the CUDA backend |
| `--openblas {auto,download,off}` / `--no-blas` | OpenBLAS for ggml's CPU backend (`auto` = system copy, else fetch into `nc/`) |
| `--cpu-baseline {avx2,avx,sse42}` | Oldest x86-64 CPU the result must run on (default `avx2`, ≈2013+) |
| `--cuda-arch LIST` / `--hip-arch LIST` | Override the arch lists below, e.g. `--cuda-arch "75-real;86-real"` |

---

## 3. GPU architecture entries: codenames and CUDA ranges

Every CUDA build passes an explicit `CMAKE_CUDA_ARCHITECTURES`. Entries are
mostly `-virtual` (PTX, JIT-compiled by the driver on first load, forward
compatible — one PTX target covers every newer card) with `-real` (machine
code) for the architectures people actually have.

| Entry | Codename | Example cards | CUDA range |
|---|---|---|---|
| `50-virtual` | Maxwell (GM10x) | GTX 750 Ti / 900 series, Tesla M60 | 6.5 – 12.x (**dropped in 13.0**) |
| `61-virtual` | Pascal (GP10x) | GTX 10 series, Tesla P100 / P40 | 8.0 – 12.x (**dropped in 13.0**) |
| `70-virtual` | Volta (GV100) | Tesla V100, Titan V | 9.0 – 12.x (**dropped in 13.0**) |
| `75-real` | Turing (TU10x) | RTX 20 series, GTX 16 series, Tesla T4 | 10.0 – current (13.x floor) |
| `80-virtual` | Ampere datacenter (GA100) | A100 / A30 | 11.0 – current |
| `86-real` | Ampere consumer (GA10x) | RTX 30 series, A40 | 11.1 – current |
| `89-real` | Ada Lovelace (AD10x) | RTX 40 series, L40 | 11.8 – current |
| `90-virtual` | Hopper (GH100/GH200) | H100 / H200 | arch since 11.8, buildable since 12.x (see §5) |
| `100` / `103` | Blackwell datacenter | B100 / B200 / B300, GB200 | 12.x – current |
| `120a-real` | Blackwell consumer | RTX 50 series | 12.8 – current |
| `121a-real` | Blackwell refresh | newer 50-series SKUs | 12.9 – current |
| `107` | Rubin (**preview**) | Vera Rubin | 13.4+ (preview, GA later) |

Reference: NVIDIA's [CUDA Toolkit, Driver, and Architecture
Matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html),
Table 3 (First/Last Toolkit columns); per-release arch sets at
https://gpucompat.com/cuda/. Latest release at time of writing: 13.4.x.

### Toolkit coverage: what 13.0 and 13.2 support

| Toolkit | Arch set (`sm_XX`) | Floor / notes |
|---|---|---|
| 13.0 (Aug 2025) | 75, 80, 86, 89, 90, 100, 103, 120, 121 | **Drops everything below 7.5** (Maxwell/Pascal/Volta gone from offline compilation); min driver R580; GCC 15 support |
| 13.1 (Dec 2025) | same as 13.0 | bugfix release, no arch changes |
| 13.2 (Mar 2026) | same as 13.1 | `cu132` wired in `pyproject.toml`, still experimental upstream; compiler/CUBLAS correctness fixes over 13.0–13.1 |
| 13.3 (May 2026) | same as 13.2 | CUDA-graph recapture improvements |
| 13.4 (Jul–Sep 2026) | above + **`107` (Rubin, preview)** | Windows-on-Arm support; Rubin GA in a later toolkit |

So for this repo: `cu130` (13.0) is the current stable CUDA build and covers
Turing through Blackwell; `cu132` compiles the identical arch list
(`75-real … 121a-real`) — 13.2 adds no new architectures over 13.0, only
toolchain fixes. Pre-Turing cards (Maxwell/Pascal/Volta) can only use
`cu118`/`cu126` builds, which is why those rows stay in the matrix.

What `cuda_architectures()` (`packaging/llamacpp.py:473-503`) generates from that table:

* CUDA 11.x keeps `50-virtual` (deprecated loudly in 12.x, but an 11.x build
  exists for old cards) and drops it from 12.x on.
* CUDA 13 drops everything below 75 — asking for 50 there is a hard nvcc
  error.
* `120a`/`121a` arrive with 12.8/12.9. They must be `-real`: family-specific
  targets are not forward compatible, so PTX from 9.0 is what covers anything
  newer.
* `filter_cuda_architectures()` then asks the installed nvcc
  (`--list-gpu-arch`) what it knows and drops anything unknown — so a stale
  table costs a slightly smaller binary, never a failed build. (It cannot
  save a build from missing *headers*, only from unknown *archs*; see §5.)

---

## 4. Host compilers: what nvcc accepts

nvcc compiles no host code itself; it hands it to the system compiler and
refuses one newer than its toolkit knew about. The CI matrix pins one GCC
per toolkit (one step under the ceiling, matching
`packaging/build_with_docker.py`):

| Toolkit | GCC | Runner |
|---|---|---|
| 11.8 | `gcc-11 / g++-11` | ubuntu-22.04 (NVIDIA's 11.8 repos stop there) |
| 12.6 / 12.9 | `gcc-13 / g++-13` | ubuntu-24.04 |
| 13.0 / 13.2 | `gcc-14 / g++-14` (13.x also accepts GCC 15) | ubuntu-24.04 |

On Windows nvcc drives MSVC only (never MinGW). Each toolkit has an
`_MSC_VER` ceiling read out of its own `crt/host_config.h` — e.g. 11.8 stops
at 1940, so the build picks toolset 14.29 when the runner ships a newer one.
A machine with only an incompatible pair (CUDA 11.8 + MSVC 14.44, which wants
CUDA ≥ 12.4 from the other direction) cannot build CUDA at all until one
side changes; the probe `.cu` compile in `_cuda_host_compiler()` proves the
pair works before CMake ever runs.

---

## 5. Known version floor: CUDA 11.8 vs PDL (the `cu118-gguf-linux` failure)

llama-cpp-python 0.3.35 (ggml 0.20.0) calls
`cudaGridDependencySynchronize()` /
`cudaTriggerProgrammaticLaunchCompletion()` in
`ggml/src/ggml-cuda/common.cuh`. Its own guard:

```c
#if (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) && CUDART_VERSION >= 11080))
#    define GGML_CUDA_USE_PDL
```

* Linux + GCC + CUDA 11.8 → PDL **on** → device code for arch 90 calls
  functions the 11.8 headers don't declare → `acc.cu` fails, whole build
  fails. Any `90` target, real or PTX, triggers it.
* Windows + MSVC + CUDA 11.8 → PDL **off** (the MSVC carve-out) → the same
  arch list compiles to no-ops → build passes, minus that optimization.

Hence `cuda_architectures()` emits `89-real` for 11.8 and only adds
`90-virtual` from CUDA 12.0 on. Hopper cards on an 11.8 build still run via
the 89-PTX fallback. Upstream's own 11.8 wheels cover 6.0–8.9 for the same
reason. If a future llama-cpp moves its minimum toolkit again, this gate
(`packaging/llamacpp.py:497`) — not the arch filter — is where to move it.

---

## 6. OpenBLAS: the CPU backend behind every backend

A GPU backend never replaces ggml's CPU backend; it sits beside it and runs
whatever the device has no kernel for. So every device but Metal wants BLAS:

* `auto` (default): use the system OpenBLAS if usable, else fetch into `nc/`
  without asking (Windows release zip / `scipy-openblas32` wheel on Linux).
* `download`: same fetch, explicitly.
* `off` / `--no-blas`: build without it — slower prompt processing, nothing
  else changes. Never interactive: CI must work unattended.

Linux note: the distro package hides headers in a multiarch subdirectory, so
CI symlinks them into one prefix and names it via `OPENBLAS_HOME` — the
layout `packaging/llamacpp.py` looks for. `python
packaging/llamacpp.py --find-openblas` reports what would be used.

---

## 7. Inspecting and installing by hand

```bash
# What would this device build with (no toolchain needed for the answer)?
python packaging/llamacpp.py --device cu129 --print-args
python packaging/llamacpp.py --device cpu --openblas auto --print-args

# What OpenBLAS is here?
python packaging/llamacpp.py --find-openblas

# Install into an environment (note --no-cache-dir is mandatory, see §1):
python packaging/llamacpp.py --device cu129 --install --python .venv/bin/python
CMAKE_ARGS="<printed line>" uv pip install --no-cache-dir llama-cpp-python==0.3.35
```

`--print-args` answers even on a machine that couldn't build the result
(exit 1 with the reason first, then the unfiltered args). `verify()` afterwards
checks the install actually built what was asked: CUDA backend present for
CUDA devices, `lib/` layout intact, linked runtimes shippable.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `common.cuh: error: cudaGridDependencySynchronize is undefined` | Toolkit too old for this llama-cpp (see §5) | Build that device with `--gguf-device cpu`, or a newer toolkit |
| nvcc refuses the host compiler (`unsupported GNU/MSVC version`) | Wrong GCC/MSVC for the toolkit (§4) | Install the table's compiler; the build picks versioned `g++-N` itself |
| `cblas_sgemm was not declared` deep in ggml | System OpenBLAS missing/misdetected | `apt install libopenblas-dev`, or `--openblas download` |
| CUDA build fails on MinGW (Windows) | Expected — nvcc only drives MSVC | Visual Studio Build Tools, "Desktop development with C++" |
| Rebuild behaves oddly | Stale `packaging/build/` or reused venv compiled for another device | `--clean`; installs always `--no-cache-dir` |
| GGUF problems you don't need | — | `--no-gguf` builds everything else |
