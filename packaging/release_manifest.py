"""Validate build outputs and write the release's machine-readable download index.

Uses only the standard library; archives are hashed in order without joining GBs
of parts on the release runner. Build-time checksums are mandatory: continuity
alone cannot detect a missing final part.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

CHUNK_SIZE = 1024 * 1024
PLATFORMS = {
    "windows-x64": ("windows", "x86_64", "zip"),
    "linux-x64": ("linux", "x86_64", "tar.gz"),
    "macos-arm64": ("macos", "arm64", "tar.gz"),
}
_DIGEST = r"[0-9a-f]{64}"
_CHECKSUM = re.compile(rf"^({_DIGEST})  (\S+)$")
_REJOINED = re.compile(rf"^# (\S+) \(rejoined\)  ({_DIGEST})$")


class ManifestError(ValueError):
    """The release would advertise an incomplete or inconsistent download."""


def _filename(name: str) -> str:
    if not re.fullmatch(r"fox-reader-[A-Za-z0-9._-]+", name):
        raise ManifestError(f"Unsafe or unexpected release filename: {name!r}")
    return name


def read_build_checksums(directory: Path) -> tuple[dict[str, str], dict[str, str]]:
    files: dict[str, str] = {}
    joined: dict[str, str] = {}
    manifests = sorted(directory.rglob("SHA256SUMS.txt"))
    if not manifests:
        raise ManifestError("No build-time checksum manifests were downloaded")
    for path in manifests:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            match = _REJOINED.fullmatch(line)
            if match:
                name, digest = match.groups()
                target = joined
            else:
                match = _CHECKSUM.fullmatch(line)
                if not match:
                    raise ManifestError(f"Invalid checksum record in {path}: {line!r}")
                digest, name = match.groups()
                target = files
            _filename(name)
            if name in target:
                raise ManifestError(f"Duplicate build checksum for {name}")
            target[name] = digest
    if not files:
        raise ManifestError("Build checksum manifests contain no assets")
    return files, joined


def publication_date(value: str | None = None) -> str:
    moment = datetime.now(timezone.utc) if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ManifestError("Publication date must include its timezone")
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_build_inventory(checksums: Path, jobs_pages: list, artifacts_pages: list) -> None:
    """Catch an entirely missing variant, even when its archive and hashes vanished."""
    jobs = {}
    for page in jobs_pages:
        for job in page["jobs"]:
            previous = jobs.get(job["name"])
            if previous is None or job["id"] > previous["id"]:
                jobs[job["name"]] = job
    expected_builds = 0
    for job in jobs.values():
        upload_steps = [step for step in job.get("steps", []) if step["name"] == "Upload checksum manifest"]
        if upload_steps:
            if job.get("conclusion") != "success" or any(step["conclusion"] != "success" for step in upload_steps):
                raise ManifestError(f"Build checksum upload did not complete: {job['name']}")
            expected_builds += 1
    if not expected_builds:
        raise ManifestError("The build run contains no completed checksum-producing jobs")
    assets = [asset for page in artifacts_pages for asset in page["artifacts"]
              if asset["name"].startswith(("checksums-", "parts-", "fox-reader-"))]
    names = {asset["name"] for asset in assets}
    if len(names) != len(assets) or any(asset.get("expired") for asset in assets):
        raise ManifestError("Build artifacts are expired or duplicated; run a fresh complete build")
    checksum_names = {name for name in names if name.startswith("checksums-")}
    downloaded = {path.parent.name for path in checksums.rglob("SHA256SUMS.txt")}
    if (len(checksum_names) != expected_builds or downloaded != checksum_names
            or len(names) != 2 * expected_builds):
        raise ManifestError("Not every completed build has both its download and checksum artifact")


def generate_manifest(artifacts: Path, checksums: Path, repository: str, tag: str,
                      notes: str = "", pub_date: str | None = None,
                      required_platforms: tuple[str, ...] = ()) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ManifestError("Expected a GitHub owner/repository")
    if not re.fullmatch(r"v?[0-9][A-Za-z0-9.+_-]*", tag):
        raise ManifestError("Release tag must be a version, optionally prefixed with v")
    version = tag.removeprefix("v")
    if len(notes) > 128 * 1024:
        raise ManifestError("Release notes exceed the updater's 128 KiB limit")
    archive_version = re.sub(r"[^A-Za-z0-9._-]+", "-", version)
    pattern = re.compile(
        rf"^fox-reader-v{re.escape(archive_version)}-"
        r"(?P<variant>(?P<device>[a-z][a-z0-9._]*)(?:-gguf(?:-(?P<gguf>[a-z][a-z0-9._]*))?)?)"
        r"-(?P<platform>windows-x64|linux-x64|macos-arm64)"
        r"\.(?P<format>zip|tar\.gz)(?:\.(?P<part>[0-9]{3}))?$"
    )
    expected, expected_joined = read_build_checksums(checksums)
    actual = {p.name: p for p in artifacts.iterdir() if p.name.startswith("fox-reader-")}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise ManifestError(f"Release assets differ from build checksums; missing={missing}, unexpected={unexpected}")
    groups: dict[str, list[tuple[Path, re.Match]]] = {}
    for name, path in actual.items():
        match = pattern.fullmatch(name)
        if not match or not path.is_file() or path.is_symlink():
            raise ManifestError(f"Unexpected asset, platform or version: {name}")
        if match["format"] != PLATFORMS[match["platform"]][2]:
            raise ManifestError(f"Wrong archive format for platform: {name}")
        whole_name = name.rsplit(".", 1)[0] if match["part"] else name
        groups.setdefault(whole_name, []).append((path, match))

    base_url = f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}"
    platforms: dict = {}
    used_joined: set[str] = set()
    for name, members in sorted(groups.items()):
        members.sort(key=lambda member: int(member[1]["part"] or 0))
        match = members[0][1]
        if match["part"]:
            numbers = [int(item[1]["part"]) for item in members]
            if numbers != list(range(1, len(members) + 1)):
                raise ManifestError(f"Split parts must be contiguous from .001: {name}")
            if name not in expected_joined:
                raise ManifestError(f"Missing build-time joined checksum: {name}")
            used_joined.add(name)
        elif len(members) != 1:
            raise ManifestError(f"Both whole and split archives were supplied: {name}")
        digest = hashlib.sha256()
        parts = []
        total = 0
        for path, _ in members:
            part_digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while block := stream.read(CHUNK_SIZE):
                    digest.update(block)
                    part_digest.update(block)
                    size += len(block)
            if not size or part_digest.hexdigest() != expected[path.name]:
                raise ManifestError(f"Empty or corrupt release asset: {path.name}")
            parts.append({"name": path.name, "url": f"{base_url}/{quote(path.name, safe='')}",
                          "size": size, "sha256": part_digest.hexdigest()})
            total += size
        if match["part"] and digest.hexdigest() != expected_joined[name]:
            raise ManifestError(f"Joined archive differs from the build: {name}")
        os_name, arch, fmt = PLATFORMS[match["platform"]]
        variants = platforms.setdefault(os_name, {}).setdefault(arch, {"variants": {}})["variants"]
        if match["variant"] in variants:
            raise ManifestError(f"Duplicate platform variant: {name}")
        variants[match["variant"]] = {
            "device": match["device"],
            "gguf": (match["gguf"] or match["device"]) if "-gguf" in match["variant"] else None,
            "archive": {"name": name, "format": fmt, "size": total, "sha256": digest.hexdigest()},
            "parts": parts,
        }
    if used_joined != expected_joined.keys():
        raise ManifestError("Build checksums refer to missing or unsplit joined archives")
    if missing_platforms := set(required_platforms) - platforms.keys():
        raise ManifestError(f"Missing required platforms: {', '.join(sorted(missing_platforms))}")
    return {"schema_version": 1, "version": version, "tag": tag, "notes": notes,
            "pub_date": publication_date(pub_date),
            "release_url": f"https://github.com/{repository}/releases/tag/{quote(tag, safe='')}",
            "platform": platforms}


def write_release_files(manifest: dict, output: Path) -> None:
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    lines = []
    for architectures in manifest["platform"].values():
        for architecture in architectures.values():
            for variant in architecture["variants"].values():
                archive = variant["archive"]
                if variant["parts"][0]["name"] != archive["name"]:
                    lines.append(f"# {archive['name']} (rejoined)  {archive['sha256']}")
                lines.extend(f"{part['sha256']}  {part['name']}" for part in variant["parts"])
    (output.parent / "SHA256SUMS.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8", newline="\n")


def verify_uploaded_assets(manifest_path: Path, remote_assets: list) -> None:
    """Compare the draft's complete inventory before making it public."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {}
    for architectures in manifest["platform"].values():
        for architecture in architectures.values():
            for variant in architecture["variants"].values():
                expected.update((part["name"], part) for part in variant["parts"])
    for path in (manifest_path, manifest_path.parent / "SHA256SUMS.txt"):
        data = path.read_bytes()
        expected[path.name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    # gh api --paginate --slurp yields one array per page.
    assets = [asset for page in remote_assets for asset in (page if isinstance(page, list) else [page])]
    actual = {asset["name"]: asset for asset in assets}
    if len(actual) != len(assets) or actual.keys() != expected.keys():
        raise ManifestError("Uploaded assets do not match this release's complete inventory")
    for name, target in expected.items():
        asset = actual[name]
        if asset.get("state") != "uploaded" or asset.get("size") != target["size"]:
            raise ManifestError(f"Incomplete uploaded asset: {name}")
        # GitHub provides digests for new uploads. Do not publish if it cannot
        # prove the uploaded bytes match the locally verified build output.
        if asset.get("digest") != f"sha256:{target['sha256']}":
            raise ManifestError(f"Missing or mismatched uploaded SHA256: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--artifacts", required=True, type=Path)
    generate.add_argument("--checksums", required=True, type=Path)
    generate.add_argument("--repository", required=True)
    generate.add_argument("--tag", required=True)
    generate.add_argument("--notes-file", required=True, type=Path)
    generate.add_argument("--pub-date")
    generate.add_argument("--require-platform", action="append", default=[])
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--build-jobs-json", type=Path)
    generate.add_argument("--build-artifacts-json", type=Path)
    verify = commands.add_parser("verify-assets")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--assets-json", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "verify-assets":
            verify_uploaded_assets(args.manifest, json.loads(args.assets_json.read_text(encoding="utf-8")))
        else:
            if bool(args.build_jobs_json) != bool(args.build_artifacts_json):
                raise ManifestError("Supply both GitHub build inventory files together")
            if args.build_jobs_json:
                validate_build_inventory(args.checksums,
                                         json.loads(args.build_jobs_json.read_text(encoding="utf-8")),
                                         json.loads(args.build_artifacts_json.read_text(encoding="utf-8")))
            manifest = generate_manifest(args.artifacts, args.checksums, args.repository, args.tag,
                                         args.notes_file.read_text(encoding="utf-8"), args.pub_date,
                                         tuple(args.require_platform))
            write_release_files(manifest, args.output)
    except (ManifestError, OSError, ValueError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")


if __name__ == "__main__":
    main()
