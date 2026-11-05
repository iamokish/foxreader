"""Release download indices must prove complete build outputs, including split tails."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "packaging" / "release_manifest.py"
spec = importlib.util.spec_from_file_location("fox_release_manifest_tests", SOURCE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def downloads(tmp_path):
    artifacts = tmp_path / "artifacts"
    checksums = tmp_path / "checksums"
    artifacts.mkdir()
    checksums.mkdir()

    def add(name, data, parts=None):
        lines = []
        if parts:
            lines.append(f"# {name} (rejoined)  {digest(data)}")
            for index, block in enumerate(parts, 1):
                part_name = f"{name}.{index:03d}"
                (artifacts / part_name).write_bytes(block)
                lines.append(f"{digest(block)}  {part_name}")
        else:
            (artifacts / name).write_bytes(data)
            lines.append(f"{digest(data)}  {name}")
        folder = checksums / name
        folder.mkdir()
        (folder / "SHA256SUMS.txt").write_text("\n".join(lines), encoding="utf-8")
    return artifacts, checksums, add


def make(artifacts, checksums, **kwargs):
    return release.generate_manifest(artifacts, checksums, "owner/repo", "v1.2.3", **kwargs)


def test_all_platforms_and_variants_with_ordered_verified_parts(downloads):
    artifacts, checksums, add = downloads
    add("fox-reader-v1.2.3-cpu-windows-x64.zip", b"win")
    add("fox-reader-v1.2.3-cpu-gguf-cu130-linux-x64.tar.gz", b"12345", [b"12", b"34", b"5"])
    add("fox-reader-v1.2.3-macos-gguf-macos-arm64.tar.gz", b"mac")
    manifest = make(artifacts, checksums, notes="New reader!", pub_date="2026-09-22T00:44:18.046Z",
                    required_platforms=("windows", "linux", "macos"))
    assert manifest["version"] == "1.2.3"
    assert manifest["notes"] == "New reader!"
    assert manifest["pub_date"] == "2026-09-22T00:44:18.046Z"
    linux = manifest["platform"]["linux"]["x86_64"]["variants"]["cpu-gguf-cu130"]
    assert linux["device"] == "cpu" and linux["gguf"] == "cu130"
    assert linux["archive"]["sha256"] == digest(b"12345")
    assert linux["archive"]["size"] == 5
    assert [part["name"][-3:] for part in linux["parts"]] == ["001", "002", "003"]
    assert linux["parts"][0]["url"].startswith("https://github.com/owner/repo/releases/download/v1.2.3/")
    win = manifest["platform"]["windows"]["x86_64"]["variants"]["cpu"]
    assert win["gguf"] is None and len(win["parts"]) == 1
    release.write_release_files(manifest, artifacts / "latest.json")
    assert json.loads((artifacts / "latest.json").read_text()) == manifest
    assert "(rejoined)" in (artifacts / "SHA256SUMS.txt").read_text()


@pytest.mark.parametrize("mutation", ["missing_first", "missing_middle", "missing_tail", "corrupt", "empty"])
def test_missing_and_corrupt_parts_are_never_advertised(downloads, mutation):
    artifacts, checksums, add = downloads
    name = "fox-reader-v1.2.3-cu130-gguf-windows-x64.zip"
    add(name, b"123456", [b"12", b"34", b"56"])
    if mutation.startswith("missing"):
        number = {"missing_first": 1, "missing_middle": 2, "missing_tail": 3}[mutation]
        (artifacts / f"{name}.{number:03d}").unlink()
    else:
        (artifacts / f"{name}.002").write_bytes(b"xx" if mutation == "corrupt" else b"")
    with pytest.raises(release.ManifestError):
        make(artifacts, checksums)


def test_joined_digest_detects_truncation_even_with_individual_hashes(downloads):
    artifacts, checksums, add = downloads
    add("fox-reader-v1.2.3-cpu-linux-x64.tar.gz", b"123456", [b"12", b"34"])
    with pytest.raises(release.ManifestError, match="Joined archive"):
        make(artifacts, checksums)


def test_rejects_mismatched_version_and_missing_platform(downloads):
    artifacts, checksums, add = downloads
    add("fox-reader-v1.2.4-cpu-linux-x64.tar.gz", b"linux")
    with pytest.raises(release.ManifestError, match="version"):
        make(artifacts, checksums)


def test_mandatory_platforms(downloads):
    artifacts, checksums, add = downloads
    add("fox-reader-v1.2.3-cpu-linux-x64.tar.gz", b"linux")
    with pytest.raises(release.ManifestError, match="Missing required platforms"):
        make(artifacts, checksums, required_platforms=("windows",))


def test_rejects_checksum_paths_outside_artifacts(downloads):
    artifacts, checksums, _ = downloads
    (checksums / "SHA256SUMS.txt").write_text(f"{digest(b'x')}  ../outside.zip")
    with pytest.raises(release.ManifestError, match="filename"):
        make(artifacts, checksums)


def test_uploaded_inventory_includes_manifest_and_requires_matching_digests(downloads):
    artifacts, checksums, add = downloads
    add("fox-reader-v1.2.3-cpu-windows-x64.zip", b"win")
    manifest = make(artifacts, checksums)
    path = artifacts / "latest.json"
    release.write_release_files(manifest, path)
    remote = [{"name": p.name, "size": p.stat().st_size, "state": "uploaded",
               "digest": f"sha256:{digest(p.read_bytes())}"} for p in artifacts.iterdir()]
    release.verify_uploaded_assets(path, [remote])
    remote[0]["digest"] = None
    with pytest.raises(release.ManifestError, match="SHA256"):
        release.verify_uploaded_assets(path, [remote])


def test_split_and_unsplit_collision_is_rejected(downloads):
    artifacts, checksums, add = downloads
    name = "fox-reader-v1.2.3-cpu-windows-x64.zip"
    add(name, b"12", [b"1", b"2"])
    (artifacts / name).write_bytes(b"12")
    with (checksums / name / "SHA256SUMS.txt").open("a") as handle:
        handle.write(f"\n{digest(b'12')}  {name}")
    with pytest.raises(release.ManifestError, match="Both whole and split"):
        make(artifacts, checksums)


def test_inventory_detects_a_whole_variant_missing_with_its_checksums(tmp_path):
    checksums = tmp_path / "checksums"
    (checksums / "checksums-cpu").mkdir(parents=True)
    (checksums / "checksums-cpu" / "SHA256SUMS.txt").write_text("unused")
    jobs = [{"jobs": [
        {"id": 1, "name": "cpu", "conclusion": "success", "steps": [
            {"name": "Upload checksum manifest", "conclusion": "success"}]},
        {"id": 2, "name": "cuda", "conclusion": "success", "steps": [
            {"name": "Upload checksum manifest", "conclusion": "success"}]},
    ]}]
    artifacts = [{"artifacts": [{"name": "checksums-cpu", "expired": False},
                                 {"name": "parts-cpu", "expired": False}]}]
    with pytest.raises(release.ManifestError, match="Not every completed build"):
        release.validate_build_inventory(checksums, jobs, artifacts)
    jobs[0]["jobs"].pop()
    release.validate_build_inventory(checksums, jobs, artifacts)
    artifacts[0]["artifacts"][0]["expired"] = True
    with pytest.raises(release.ManifestError, match="expired"):
        release.validate_build_inventory(checksums, jobs, artifacts)


def test_inventory_uses_the_latest_attempt_for_each_job(tmp_path):
    (tmp_path / "checksums-cpu").mkdir()
    (tmp_path / "checksums-cpu" / "SHA256SUMS.txt").write_text("unused")
    jobs = [{"jobs": [
        {"id": 1, "name": "cpu", "conclusion": "failure", "steps": [
            {"name": "Upload checksum manifest", "conclusion": "skipped"}]},
        {"id": 2, "name": "cpu", "conclusion": "success", "steps": [
            {"name": "Upload checksum manifest", "conclusion": "success"}]},
    ]}]
    artifacts = [{"artifacts": [{"name": "checksums-cpu", "expired": False},
                                 {"name": "parts-cpu", "expired": False}]}]
    release.validate_build_inventory(tmp_path, jobs, artifacts)
