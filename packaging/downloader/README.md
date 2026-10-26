# Fox Reader Downloader & Installer

A standalone graphical installer that automates downloading, verifying, rejoining split parts, and extracting Fox Reader releases.

## Features

- **Hardware Detection**: Probes OS, architecture, and NVIDIA GPU/CUDA driver version (`cu130`, `cu129`, `cu126`, `cu118`, or `cpu`).
- **Auto-Recommendations**: Matches detected hardware to the optimal available GitHub release build, recommending the AI translation backend (GGUF) by default.
- **Split-File Rejoiner**: Automatically downloads multi-part archives (`.001`, `.002`, ...), verifies SHA256 checksums from `SHA256SUMS.txt`, and concatenates them.
- **Portable Installation**: Defaults to `./FoxReader/` alongside the downloader, with a folder browser.
- **Auto-Cleanup**: Deletes temporary downloaded parts and archives upon extraction.
- **Informational Completion**: Shows a clear completion status with the destination folder.

## Running Locally

To run the downloader from source:

```bash
uv pip install -e ".[downloader]"
python packaging/downloader/main.py
```

## Compiling with Nuitka

To compile into a standalone single-file binary:

```bash
python packaging/downloader/build_downloader.py --out-dir dist
```

This produces:

- `dist/fox-downloader.exe` on Windows.
- `dist/fox-downloader` on Linux.
