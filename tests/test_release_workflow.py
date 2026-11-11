"""Exercise release shell steps against draft-aware GitHub CLI responses."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
DRAFT_URL = "https://github.com/owner/repo/releases/tag/untagged-draft"

# The tag REST endpoint deliberately fails, as it did for the 0.3.0 draft.
# Other calls are restricted to the CLI operations used to prepare a draft.
FAKE_GH = r'''
gh() {
  printf '%s\t' "$@" >> gh-calls.txt
  printf '\n' >> gh-calls.txt
  if [[ "$1 $2" == "release view" ]]; then
    [[ "$(<release-state.txt)" != missing ]] || return 1
    case "${5:-}" in
      isDraft)
        if [[ "$(<release-state.txt)" == draft ]]; then echo true; else echo false; fi ;;
      databaseId) printf '%s\n' "$FAKE_RELEASE_ID" ;;
      url) printf '%s\n' "$FAKE_DRAFT_URL" ;;
      "") printf '%s\n' "$FAKE_DRAFT_URL" ;;
      *) echo "Unexpected release view: $*" >&2; return 2 ;;
    esac
  elif [[ "$1 $2" == "release create" || "$1 $2" == "release edit" ]]; then
    if [[ "$2" == create ]]; then echo published > release-state.txt; fi
    for option in "$@"; do
      case "$option" in
        --draft|--draft=true) echo draft > release-state.txt ;;
        --draft=false) echo published > release-state.txt ;;
        --prerelease|--prerelease=true) echo true > prerelease-state.txt ;;
        --prerelease=false) echo false > prerelease-state.txt ;;
        --latest=false) echo false > latest-state.txt ;;
      esac
    done
  elif [[ "$1" == api ]]; then
    for argument in "$@"; do
      case "$argument" in
        repos/owner/repo/releases/tags/*)
          echo 'gh: Not Found (HTTP 404)' >&2; return 1 ;;
        'repos/owner/repo/releases/123456/assets?per_page=100')
          cat remote-assets.json; return 0 ;;
      esac
    done
    echo "Unexpected GitHub API request: $*" >&2; return 2
  else
    echo "Unexpected gh invocation: $*" >&2; return 2
  fi
}
python() { "$TEST_PYTHON" "$@"; }
'''


@pytest.fixture
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture
def runner(tmp_path, workflow):
    bash = shutil.which("bash")
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if git_bash.is_file():
            bash = str(git_bash)
    if not bash:
        pytest.skip("Release workflow shell checks require Bash")

    packaging = tmp_path / "packaging"
    packaging.mkdir()
    shutil.copyfile(ROOT / "packaging/release_manifest.py", packaging / "release_manifest.py")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    name = "fox-reader-v0.3.0-cpu-linux-x64.tar.gz"
    contents = b"verified build output"
    digest = hashlib.sha256(contents).hexdigest()
    (artifacts / name).write_bytes(contents)
    (artifacts / "SHA256SUMS.txt").write_text(f"{digest}  {name}\n", encoding="utf-8")
    manifest = {"platform": {"linux": {"x86_64": {"variants": {
        "cpu": {"parts": [{"name": name, "size": len(contents), "sha256": digest}]},
    }}}}}
    (artifacts / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    remote_assets = [
        {"name": path.name, "state": "uploaded", "size": path.stat().st_size,
         "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in artifacts.iterdir()
    ]
    (tmp_path / "remote-assets.json").write_text(json.dumps([remote_assets]), encoding="utf-8")
    (tmp_path / "complete_release_body.md").write_text("Release notes\n", encoding="utf-8")
    (tmp_path / "release-state.txt").write_text("draft", encoding="utf-8")
    (tmp_path / "prerelease-state.txt").write_text("true", encoding="utf-8")
    (tmp_path / "latest-state.txt").write_text("auto", encoding="utf-8")
    (tmp_path / "workflow-output.txt").touch()
    (tmp_path / "summary.md").touch()

    def run(step_name, *, tag="0.3.0", release_id="123456"):
        step = next(step for step in workflow["jobs"]["release"]["steps"]
                    if step["name"] == step_name)
        outputs = dict(line.split("=", 1) for line in
                       (tmp_path / "workflow-output.txt").read_text().splitlines())
        expressions = {
            "${{ secrets.GITHUB_TOKEN }}": "test-token",
            "${{ steps.target.outputs.tag }}": tag,
            "${{ github.repository }}": "owner/repo",
            "${{ steps.release.outputs.release_id }}": outputs.get("release_id", ""),
        }
        env = os.environ.copy()
        env.update({
            "GITHUB_OUTPUT": "workflow-output.txt",
            "GITHUB_STEP_SUMMARY": "summary.md",
            "FAKE_RELEASE_ID": release_id,
            "FAKE_DRAFT_URL": DRAFT_URL,
            "TEST_PYTHON": Path(sys.executable).as_posix(),
        })
        env.update({key: expressions.get(value, value) for key, value in step.get("env", {}).items()})
        script = tmp_path / "step.sh"
        script.write_text(FAKE_GH + "\n" + step["run"], encoding="utf-8", newline="\n")
        return subprocess.run([bash, "--noprofile", "--norc", "step.sh"], cwd=tmp_path,
                              env=env, capture_output=True, text=True, timeout=30)

    return run


def calls(tmp_path):
    return [line.rstrip("\t").split("\t") for line in
            (tmp_path / "gh-calls.txt").read_text().splitlines()]


@pytest.mark.parametrize("initial_state", ["missing", "draft"])
@pytest.mark.parametrize("tag,prerelease", [("0.3.0", False), ("v0.3.0", False),
                                            ("v0.4.0-rc.1", True)])
def test_new_and_existing_releases_remain_verified_drafts(runner, tmp_path, initial_state, tag, prerelease):
    (tmp_path / "release-state.txt").write_text(initial_state)
    result = runner("Create or resume draft release", tag=tag)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "workflow-output.txt").read_text().strip() == "release_id=123456"

    result = runner("Verify uploaded assets and leave draft for review", tag=tag)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "release-state.txt").read_text().strip() == "draft"
    assert (tmp_path / "prerelease-state.txt").read_text().strip() == str(prerelease).lower()
    if prerelease:
        assert (tmp_path / "latest-state.txt").read_text().strip() == "false"
    assert DRAFT_URL in (tmp_path / "summary.md").read_text()
    assert any("repos/owner/repo/releases/123456/assets?per_page=100" in call for call in calls(tmp_path))
    mutations = [call[1] for call in calls(tmp_path) if call[:2] in
                 (["release", "create"], ["release", "edit"])]
    assert mutations == ["create" if initial_state == "missing" else "edit"]


def test_published_release_is_never_modified(runner, tmp_path):
    (tmp_path / "release-state.txt").write_text("published")
    result = runner("Create or resume draft release")
    assert result.returncode != 0
    assert "immutable" in result.stderr
    assert all(call[:2] == ["release", "view"] for call in calls(tmp_path))
    assert (tmp_path / "workflow-output.txt").read_text() == ""


@pytest.mark.parametrize("release_id", ["", "null", "not-a-number"])
def test_invalid_draft_id_stops_before_asset_lookup(runner, tmp_path, release_id):
    result = runner("Create or resume draft release", release_id=release_id)
    assert result.returncode != 0
    assert (tmp_path / "workflow-output.txt").read_text() == ""
    assert all(call[0] != "api" for call in calls(tmp_path))


def test_corrupt_uploaded_asset_does_not_report_ready_for_review(runner, tmp_path):
    result = runner("Create or resume draft release")
    assert result.returncode == 0, result.stdout + result.stderr
    remote = json.loads((tmp_path / "remote-assets.json").read_text())
    remote[0][0]["digest"] = "sha256:" + "0" * 64
    (tmp_path / "remote-assets.json").write_text(json.dumps(remote))

    result = runner("Verify uploaded assets and leave draft for review")
    assert result.returncode != 0
    assert "mismatched uploaded SHA256" in result.stderr
    assert (tmp_path / "summary.md").read_text() == ""
    assert (tmp_path / "release-state.txt").read_text().strip() == "draft"
