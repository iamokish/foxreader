"""GitHub Releases API client and asset grouping logic for Fox Reader Downloader."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_REPO = "iamokish/foxreader"
API_URL = f"https://api.github.com/repos/{DEFAULT_REPO}/releases"
USER_AGENT = "FoxReader-Downloader/1.0"


def fetch_releases(
    repo: str = DEFAULT_REPO,
    include_prereleases: bool = False,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """Fetch published releases from GitHub API.

    Returns a list of raw release dicts sorted newest first.
    """
    url = f"https://api.github.com/repos/{repo}/releases"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github.v3+json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code == 403:
            raise RuntimeError(
                "GitHub API rate limit reached. Please wait a few minutes or check your connection."
            ) from err
        raise RuntimeError(f"GitHub API error: HTTP {err.code} - {err.reason}") from err
    except Exception as err:
        raise RuntimeError(f"Failed to connect to GitHub: {err}") from err

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response from GitHub API: {data}")

    releases = []
    for r in data:
        if r.get("draft", False):
            continue
        if not include_prereleases and r.get("prerelease", False):
            continue
        releases.append(r)

    # Fallback if no stable releases exist but prereleases do
    if not releases and data:
        releases = [r for r in data if not r.get("draft", False)]

    return releases


FILENAME_REGEX = re.compile(
    r"^fox-reader-v?(?P<version>[^-]+)-(?P<device>[^-]+?)(?:-(?P<gguf>gguf))?-(?P<platform>[a-zA-Z0-9_-]+?)\.(?P<ext>zip|tar\.gz)(?:\.(?P<part>\d{3}))?$"
)


def parse_release_assets(release_data: dict[str, Any]) -> dict[str, Any]:
    """Group split files and classify release variants."""
    tag = release_data.get("tag_name", "unknown")
    name = release_data.get("name") or tag
    published_at = release_data.get("published_at", "")
    assets = release_data.get("assets", [])

    sha256sums_url = None
    grouped: dict[str, dict[str, Any]] = {}

    for asset in assets:
        filename = asset.get("name", "")
        download_url = asset.get("browser_download_url", "")
        size = asset.get("size", 0)

        if filename == "SHA256SUMS.txt":
            sha256sums_url = download_url
            continue

        match = FILENAME_REGEX.match(filename)
        if not match:
            # Not a recognizable Fox Reader release archive
            continue

        version = match.group("version")
        device = match.group("device")
        has_gguf = bool(match.group("gguf"))
        platform_name = match.group("platform")
        ext = match.group("ext")
        part_str = match.group("part")

        # Base archive name without .001 suffix
        gguf_str = "-gguf" if has_gguf else ""
        base_name = f"fox-reader-v{version}-{device}{gguf_str}-{platform_name}.{ext}"

        if base_name not in grouped:
            # Infer OS from platform_name
            os_target = "windows" if "windows" in platform_name else "linux" if "linux" in platform_name else "other"
            grouped[base_name] = {
                "base_name": base_name,
                "version": version,
                "device": device,
                "has_gguf": has_gguf,
                "platform": platform_name,
                "os": os_target,
                "ext": ext,
                "is_split": False,
                "parts": [],
                "total_size": 0,
            }

        entry = grouped[base_name]
        if part_str:
            entry["is_split"] = True
            entry["parts"].append({
                "filename": filename,
                "url": download_url,
                "size": size,
                "part_num": int(part_str),
            })
            entry["total_size"] += size
        else:
            entry["parts"] = [{
                "filename": filename,
                "url": download_url,
                "size": size,
                "part_num": 1,
            }]
            entry["total_size"] = size

    # Sort parts for each split variant by part number
    variant_list = []
    for var in grouped.values():
        var["parts"].sort(key=lambda p: p["part_num"])
        variant_list.append(var)

    # Sort variants cleanly: OS -> Device -> GGUF
    variant_list.sort(key=lambda v: (v["os"], v["device"], not v["has_gguf"]))

    return {
        "tag": tag,
        "name": name,
        "published_at": published_at,
        "sha256sums_url": sha256sums_url,
        "variants": variant_list,
    }


def parse_sha256sums_text(text: str) -> dict[str, str]:
    """Parse SHA256SUMS.txt into a dict mapping filenames/rejoined names to checksums."""
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Rejoined entry: # fox-reader-...zip (rejoined)  <sha256>
        if line.startswith("#") and "(rejoined)" in line:
            parts = line.split()
            if len(parts) >= 4:
                archive_name = parts[1]
                digest = parts[-1].lower()
                checksums[archive_name] = digest
            continue

        # Standard entry: <sha256>  <filename>
        parts = line.split()
        if len(parts) >= 2:
            digest = parts[0].lower()
            filename = parts[1].lstrip("*")
            checksums[filename] = digest

    return checksums


def fetch_sha256sums(url: str, timeout: int = 15) -> dict[str, str]:
    """Download and parse SHA256SUMS.txt."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
        return parse_sha256sums_text(text)
    except Exception as err:
        raise RuntimeError(f"Failed to fetch SHA256 checksums from {url}: {err}") from err


def match_recommended_variant(
    variants: list[dict[str, Any]],
    hw_info: dict[str, Any],
    prefer_gguf: bool = True,
) -> dict[str, Any] | None:
    """Find the best matching variant for the detected hardware."""
    os_name = hw_info.get("os", "windows")
    target_device = hw_info.get("recommended_device", "cpu")

    # Filter to variants matching user's OS
    os_variants = [v for v in variants if v["os"] == os_name]
    if not os_variants:
        return None

    # Handle Windows cu129 special name cu129_win
    target_devices = [target_device]
    if os_name == "windows" and target_device == "cu129":
        target_devices = ["cu129_win", "cu129"]

    # Fallback ladder if specific CUDA variant is not in the release
    cuda_ladder = ["cu130", "cu129_win", "cu129", "cu126", "cu118", "cpu"]
    if target_device in cuda_ladder:
        idx = cuda_ladder.index(target_device if target_device != "cu129" or os_name != "windows" else "cu129_win")
        candidates = cuda_ladder[idx:]
    else:
        candidates = ["cpu"]

    # Try searching down the ladder
    for dev in candidates:
        matching_dev = [v for v in os_variants if v["device"] == dev]
        if not matching_dev:
            continue

        # 1. Matches device + preferred gguf
        for v in matching_dev:
            if v["has_gguf"] == prefer_gguf:
                return v

        # 2. Matches device + opposite gguf
        return matching_dev[0]

    # Fallback: first available OS variant
    return os_variants[0]
