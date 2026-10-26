# Building Fox Reader binaries yourself

What you get at the end: a folder with `launcher` (double-click to start),
which runs the app with no Python or Node.js installed on that machine.

Time and disk (rough, one CPU build):

| Build | Time | Free disk needed |
|---|---|---|
| CPU | 30–60 min | 15 GB |
| CUDA (GPU) | 45–90 min | 25 GB |

Most of the time is the computer compiling things — you just wait.

---

## 0. Pick your build

Find your situation in the table. The `--device` value is what you will
paste into the build command later.

| Your machine | `--device` | CUDA toolkit |
|---|---|---|
| No NVIDIA GPU, any OS | `cpu` | none needed |
| NVIDIA GTX 10xx / RTX 20xx | `cu118` | 11.8 |
| NVIDIA RTX 30xx / 40xx | `cu126` or `cu129` | 12.6 / 12.9 |
| NVIDIA RTX 50xx / newest cards | `cu129` or `cu130` | 12.9 / 13.0 |
| AMD GPU | — | coming soon (build `cpu` for now) |

Notes:

- On **Windows**, CUDA 12.9 uses the name `cu129_win` instead of `cu129`.
  Every other name is the same on both systems.
- The CUDA toolkit is only needed to *build* a GPU build (the translator
  backend compiles from source). To *run* a finished build you only need an
  NVIDIA driver, no toolkit.
- `cu118` on Linux wants Ubuntu 22.04 (NVIDIA's own 11.8 packages stop
  there). Everything else builds on Ubuntu 24.04.
- When in doubt, build `cpu` first. It runs everywhere and needs no toolkit.

---

## 1. Install the base tools (everyone)

You need five things: Python 3.12, Node.js, a JS package manager,
`uv`, Git, and a C compiler. Install them in this order.

### 1a. Python 3.12 (exactly 3.12 — not 3.11, not 3.13)

Download: <https://www.python.org/downloads/> (pick any **3.12.x**).

**Windows:**

1. Run the installer.
2. Tick **“Add python.exe to PATH”** on the first page, then Install.
3. Check it (PowerShell):

```powershell
python --version
# Python 3.12.x
```

**Linux (Ubuntu 24.04 already ships Python 3.12):**

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv
python3 --version
# Python 3.12.x
```

On Ubuntu 22.04, 3.12 is not in the default packages:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

From here on, if a command says `python`, Linux users type `python3`
(or set it once: `sudo apt install -y python-is-python3`).

### 1b. Node.js 24 + package manager

Download: <https://nodejs.org/en/download> (LTS, version 22 or newer;
24 is what the release builds use).

Check it (same on both systems):

```bash
node --version
# v24.x.x
npm --version
```

Then install `pnpm` (the primary manager; the repo pins pnpm 11):

```bash
npm install -g pnpm@11
pnpm --version
# 11.x.x
```

> If you prefer plain npm, every `pnpm install` below is `npm install`
> and every `pnpm build` / `pnpm run build` is `npm run build`,
> run inside the `frontend/` folder. Stick to one of them per checkout.

### 1c. uv (Python package manager used by the build)

```bash
python -m pip install uv
uv --version
```

(Linux users: replace `python` with `python3` if needed.)

### 1d. Git

Download: <https://git-scm.com/downloads> (Windows: default options are fine).

```bash
git --version
```

Linux shortcut:

```bash
sudo apt install -y git
```

### 1e. C compiler (for the tiny launcher program)

**Windows — pick one:**

- **Option A (recommended, required for GPU builds):**
  Visual Studio Build Tools: <https://visualstudio.microsoft.com/downloads/> —
  scroll to “All downloads” → “Build Tools for Visual Studio”, and in the
  installer tick the **“Desktop development with C++”** workload.
- **Option B (CPU builds only):** MinGW-w64 `gcc` on `PATH`
  (e.g. from <https://winlibs.com/> or MSYS2 `mingw-w64-x86_64-gcc`).
  With no compiler at all, the build downloads Nuitka's own MinGW
  automatically — CPU builds still work.

> Windows GPU builds must use Option A: NVIDIA's compiler only accepts
> Microsoft's compiler, never MinGW.

**Linux:**

```bash
sudo apt install -y build-essential
gcc --version
```

---

## 2. Extras for GPU builds (skip this whole section for `cpu`)

### 2a. NVIDIA driver (build machine and every run machine)

Install the latest Game Ready / Studio driver from
<https://www.nvidia.com/drivers>. Any recent driver covers CUDA 11.8
through 13.0. Check yours with:

```bash
nvidia-smi
```

### 2b. CUDA toolkit (build machine only)

Download: <https://developer.nvidia.com/cuda-downloads>
(older versions: <https://developer.nvidia.com/cuda-toolkit-archive>).
Install exactly the version matching your `--device`:

| `--device` | Toolkit to install |
|---|---|
| `cu118` | 11.8.0 |
| `cu126` | 12.6.x |
| `cu129` / `cu129_win` | 12.9.x |
| `cu130` | 13.0.x |

Check it afterwards (both systems):

```bash
nvcc --version
```

### 2c. Matching GCC (Linux GPU builds only)

NVIDIA's compiler rejects a host compiler that is too new. Install the
GCC the toolkit accepts — the build picks it up by itself:

| Toolkit | GCC to install |
|---|---|
| 11.8 | `gcc-11 g++-11` |
| 12.6 / 12.9 | `gcc-13 g++-13` |
| 13.0 | `gcc-14 g++-14` |

Example for a `cu129` build:

```bash
sudo apt install -y gcc-13 g++-13
```

### 2d. OpenBLAS (CPU backend of the translator)

- **Linux:** install the system library (recommended), or skip it and add
  `--openblas download` to the build command later:

```bash
sudo apt install -y libopenblas-dev pkg-config
```

- **Windows / macOS:** nothing to do — the build fetches it automatically.

> Skipping the translator backend entirely is also possible
> (`--no-gguf` in step 4). Then you need neither OpenBLAS nor CUDA,
> but local GGUF translation models will not load.

---

## 3. Get the code

**Windows (PowerShell):**

```powershell
cd ~
git clone https://github.com/iamokish/foxreader
cd foxreader
```

**Linux:**

```bash
cd ~
git clone https://github.com/iamokish/foxreader
cd foxreader
```

---

## 4. Build it

Everything is one script. It builds the web UI, creates an isolated
environment, compiles the translator backend, compiles the app with
Nuitka, and packs the result into `dist/`.

### CPU build (no GPU needed)

**Windows:**

```powershell
python packaging/build.py --device cpu
```

**Linux:**

```bash
python3 packaging/build.py --device cpu
```

### GPU build (examples — use your `--device` from section 0)

**Windows (CUDA 12.9):**

```powershell
python packaging/build.py --device cu129_win
```

**Linux (CUDA 12.9):**

```bash
python3 packaging/build.py --device cu129
```

**Linux (older cards, CUDA 11.8 on Ubuntu 22.04):**

```bash
python3 packaging/build.py --device cu118
```

### Useful variants (append to any command above)

```bash
# Rebuild from scratch (fixes most "it worked yesterday" states)
python packaging/build.py --device cpu --clean

# Skip the translator backend: no OpenBLAS/CUDA needed, fastest build.
# (GGUF translation models will not load in the result.)
python packaging/build.py --device cpu --no-gguf

# CUDA torch but CPU translator (GPU for OCR, CPU for translation)
python packaging/build.py --device cu129 --gguf-device cpu

# Fetch OpenBLAS automatically instead of installing it (Linux)
python3 packaging/build.py --device cpu --openblas download

# Run on older CPUs (default avx2 = 2013+ Intel/AMD; sse42 = anything 64-bit)
python packaging/build.py --device cpu --cpu-baseline sse42

# Target specific GPUs instead of the defaults (quote it: the shell eats `;`)
python packaging/build.py --device cu129 --cuda-arch "75-real;86-real"

# Split the finished archive into 1000 MB (1 GB) parts
python packaging/build.py --device cu130 --split-bin 1000
```

The finished archive lands in `dist/`, e.g.
`dist/fox-reader-v0.1.0-cpu-gguf-windows-x64.zip` /
`dist/fox-reader-v0.1.0-cpu-gguf-linux-x64.tar.gz`, next to a
`dist/SHA256SUMS.txt` listing its SHA-256.

### If the build stops with an error

1. Read the last `error:` line — it usually names the missing piece.
2. Re-run with `--clean` (a stale `packaging/build/` or venv causes most
   repeat failures).
3. See [Troubleshooting](#troubleshooting) below.

---

## 4a. Splitting a big archive (`--split-bin`)

Only needed if you are **uploading** a build somewhere with a file-size
limit. GitHub refuses a release file over 2 GB, and a CUDA build with the
translator backend in it is 3–4 GB. Splitting into 1 GB parts ensures reliable,
fast uploads.

```bash
python packaging/build.py --device cu130 --split-bin 1000
```

What it does, after the archive is built:

- **Archive under 1000 MB?** Nothing happens. The file is left exactly as
  it is and can be uploaded directly.
- **Archive over 1000 MB?** It is cut into
  `fox-reader-….zip.001`, `….zip.002`, … of at most 1000 MB each, and the
  oversized archive is **deleted** — so an upload glob cannot pick up a
  file the host will reject.

The number is in megabytes and optional: a bare `--split-bin` uses 1000.

> Needs room for one extra copy of the archive while it is being cut
> (the parts are written before the original is removed). Add 4 GB of free
> disk to the figures at the top of this page if you use it.

The parts are the archive's own bytes cut at fixed offsets, so putting them
back needs nothing but your shell:

**Windows:**

```powershell
cmd /c "copy /b fox-reader-….zip.001+fox-reader-….zip.002 fox-reader-….zip"
```

**Linux / macOS:**

```bash
cat fox-reader-….tar.gz.* > fox-reader-….tar.gz
```

Then extract the joined file as usual.

### `dist/SHA256SUMS.txt`

Written on every build, split or not, in the format `sha256sum -c` reads:

```
# fox-reader-v0.2.0-cu130-gguf-windows-x64.zip (rejoined)  3f8a…
4c1d…  fox-reader-v0.2.0-cu130-gguf-windows-x64.zip.001
9b07…  fox-reader-v0.2.0-cu130-gguf-windows-x64.zip.002
```

```bash
cd dist && sha256sum -c SHA256SUMS.txt
```

The plain lines cover the files as they are on disk. The `#` line — which
`sha256sum -c` skips — carries the digest of the **joined** archive, which
is not a file any more, so you can still check the result of rejoining:

```bash
cat fox-reader-….zip.* | sha256sum     # compare with the `#` line
```

Building more than one variant into the same `dist/` is fine: each build
updates its own lines and leaves the others alone (and drops lines for
files that are no longer there).

Releases built by GitHub Actions carry one `SHA256SUMS.txt` over every
attached file, plus a signed build attestation per file:

```bash
gh attestation verify <file> --repo iamokish/foxreader
```

---

## 5. Run your build

1. Unzip / untar the archive from `dist/` anywhere you like.
2. Start `launcher` (`launcher.exe` on Windows) — it starts the backend
   and opens the UI at `http://127.0.0.1:7954`.
3. First launch opens `/setup`, which downloads the models (needs internet
   once; weights live in `models/` afterwards).

---

## 6. All build flags

| Flag | Effect |
|---|---|
| `--device {cpu,cu118,cu126,cu129,cu129_win,cu130,rocm71,rocm72}` | Which PyTorch to bundle (default `cpu`) |
| `--no-gguf` | Skip the llama.cpp translator backend entirely |
| `--gguf-device DEVICE` | Build llama.cpp for another device than `--device` |
| `--openblas {auto,download,off}` | OpenBLAS for the CPU backend (`auto` = system, else download) |
| `--no-blas` | Same as `--openblas off` |
| `--cpu-baseline {avx2,avx,sse42}` | Oldest CPU the result must run on (default `avx2`) |
| `--cuda-arch LIST` / `--hip-arch LIST` | Override GPU architectures (e.g. `75-real;86-real`) |
| `--split-bin [MB]` | Cut the finished archive into parts of at most MB megabytes (default 1000) if it is larger; see [4a](#4a-splitting-a-big-archive---split-bin) |
| `--skip-frontend` | Reuse an already built `frontend/static` |
| `--keep-venv` | Keep the build environment for the next rebuild |
| `--clean` | Delete all previous build outputs first |
| `--deep-compile` | Also compile dependencies (hours; release-grade) |
| `--nuitka-arg ARG` | Forward one raw flag to Nuitka (repeatable) |

Environment (advanced):

| Variable | Effect |
|---|---|
| `FOX_BUILD_KEEP_DUP_LIBS=1` | Ship `bin/` as Nuitka produced it (no duplicate-`.so` slimming) |

---

## 7. AMD / ROCm / HIP

GPU builds for AMD cards are **coming soon** and are not available yet.
Until then, build `cpu` — everything except GPU-accelerated local
translation works, on any machine.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` (Linux) | Use `python3`, or `sudo apt install -y python-is-python3` |
| `Python 3.13` breaks the install | The project needs **3.12** (`pyproject.toml`: `>=3.12,<3.13`). Install 3.12 side by side and run it explicitly |
| `No module named uv` / `uv: command not found` | Re-run `python -m pip install uv`, then restart the terminal |
| `UNKNOWN ... symlink` on `pnpm install` (Windows) | Enable Developer Mode (Settings → System → For developers), or run `pnpm install --node-linker=hoisted` in `frontend/` |
| `no working C compiler found` | Windows: install VS Build Tools with “Desktop development with C++”. Linux: `sudo apt install -y build-essential`. Check with `python packaging/toolchain.py --which` |
| `nvcc: command not found` | The CUDA toolkit is not installed (or not on `PATH`). See 2b; restart the terminal after installing |
| nvcc refuses the host compiler (Linux) | Wrong GCC for the toolkit — see the 2c table (e.g. CUDA 12.9 wants `gcc-13`) |
| CUDA build fails on MinGW (Windows) | Expected: install VS Build Tools (Option A in 1e); nvcc never drives MinGW |
| `cblas_sgemm was not declared` (GGUF compile) | System OpenBLAS missing/misdetected — install `libopenblas-dev` (2d) or rebuild with `--openblas download` |
| Out of disk halfway | Free 15 GB (CPU) / 25 GB (CUDA); the same gigabytes are written three times before archiving. `--split-bin` wants one more copy of the archive on top |
| `could not split …: No space left on device` | The archive itself is untouched — free some space and re-run, or upload it by hand |
| GitHub rejects the upload: file too large | Build again with `--split-bin 1000`, or join and re-split by hand (see [4a](#4a-splitting-a-big-archive---split-bin)) |
| A downloaded `.zip.001` will not open | It is one part of a split archive. Get every part and join them first — see [4a](#4a-splitting-a-big-archive---split-bin) |
| Rebuild behaves oddly | Add `--clean` and build again |
| GGUF compile problems you don't need | Add `--no-gguf` — builds everything else, skips llama.cpp |
