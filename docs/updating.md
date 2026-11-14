# Updating Fox Reader

In a packaged build, open **Settings → Updates**, then select **Check for updates**. Checking is
explicit: opening Fox Reader does not contact GitHub. The app checks the latest
stable release and keeps the installed Windows/Linux/macOS, architecture,
CPU/CUDA, and GGUF choices. If the release lacks an exact match, use the release
page to choose a distribution yourself.

When running from source, update controls are omitted and the update API is
disabled. A copied `build-info.json` does not enable updates in a source checkout.

## Download and install

In supported packaged installations, choose **Download and install**. **Save
your work first:** Fox Reader closes when the separate **Fox Reader Updater**
window is ready. The updater waits for the backend and launcher to exit,
installs the release, and shows success. It does not restart Fox Reader. Close
the updater, then start Fox Reader yourself when ready.

If preparation fails or the updater cannot start safely, Fox Reader stays open
and reports the error. A failed handoff restores normal backend health warnings.

Downloads first go under `updates/<version>/` beside the launcher. Every part
is checked for its declared size and SHA256; split archives are joined in their
numbered order and checked again. Completed parts survive cancellation and
failed downloads, and a retry checks their hashes before reusing them. After
verification, Fox Reader extracts and validates a prepared tree beside the
installation. The native updater checks that tree against the verified file
inventory before replacing anything.

You can cancel downloading or preparation from Settings. After the handoff,
follow the updater window; cancellation is available only before file
replacement begins. Once files are being changed, let installation or rollback
finish. Settings stops retrying backend requests when the app intentionally
closes for the update.

Cancelling in the updater discards its prepared files when you close that
window. Start a new download to try again. Cancelling earlier in Settings keeps
verified downloads available for reuse.
If the updater is interrupted while cleaning up a cancelled update, the next
launcher retries removal of its remaining temporary files.

| Installation content | Update behavior |
|---|---|
| `bin/`, `legal/` | Replaced exactly by the new release; obsolete files are removed. |
| `fonts/` | Merged: packaged paths replace matching old paths; other user fonts stay. |
| Launcher, updater, icons, platform viewer libraries, desktop entry files | Refreshed from the release; obsolete managed components are removed. |
| `build-info.json` | Replaced with the new release metadata; restored to the previous version if installation rolls back. |
| `config/`, `models/` | Left untouched by installation. |

Other personal files remain in place. The normal app startup/shutdown cache
cleanup still applies.

## Failure recovery and cleanup

File replacement uses a journal and keeps the previous managed files until the
update commits. If replacement fails, the updater automatically restores those
files. If a process or computer stops during replacement or rollback, the next
launcher detects the unfinished transaction and runs recovery before starting
the backend. An active updater or an unfinished recovery blocks normal startup.

Prepared files, the verified archive, and rollback copies live temporarily in
`.foxreader-update-<id>/` beside the launcher. After success, large temporary
files are removed immediately. Closing the updater starts a silent cleanup
helper so its running copy can also be removed. If locked files prevent complete
cleanup, a later launcher retries cleanup of the completed transaction. A
failed cleanup of a completed update does not undo the update.

Keep an unfinished transaction folder until recovery completes: it holds the
files needed to restore the previous installation. The updater reports failures
instead of restarting the app automatically.

## Manual updates and configuration

If the installed updater is unavailable, **Download update** keeps the existing
manual workflow and displays the verified archive path. Older packaged
releases without `build-info.json` use the release page to select a
download. Existing releases that predate the native updater need one manual
update to a release that includes it; subsequent updates can install through
Settings. After downloading manually:

1. Close Fox Reader and extract the archive into a **new folder**.
2. Copy `config`, `fonts`, and `models` from the old installation to the new one.
   Keep the old installation until the new version works.
3. Start the new launcher and check the setup and reader.

Release archives do not include `fox_config.yaml`. The new backend creates it
from built-in defaults when absent and updates existing configuration on launch,
adding new fields and repairing invalid values while keeping valid settings
and unknown fields when the YAML parses. Repairs update the file directly
without automatic backups; malformed documents are replaced with defaults.
Copy `user_config.yaml` too if you use it; it remains the preferred configuration
file. You do not need to merge a template after updating. See
[Configuration](configuration.md) for repair behavior and diagnostics.

A split archive needs space for its parts and the joined archive. Automatic
installation also needs space for extraction, the merged fonts tree, and
recovery copies of the launchers and macOS app. Fox Reader checks free space
before preparing the update and removes numbered parts after a successful join.

Local builds made with `--no-window` can check newer versions and release notes,
but cannot download an update archive through the app: published CI archives
include native windows. Rebuild the newer release with `--no-window` to keep Fox
Reader's graphical windows excluded.

## Native updater implementation

`foxreader-updater.exe` on Windows, or `foxreader-updater` on Linux/macOS, ships
beside the launcher in native-window builds. It is a separate C++17 program
using Win32, GTK 3, or Cocoa respectively. `--no-window` builds omit it.

The app launches a transaction-local copy through an authenticated, one-use
inherited pipe. Opening the updater by itself does not authorize an install.
The handoff supplies the installation, transaction, process identities, and the
expected inventory digest; the updater accepts only its managed paths. Archive
extraction rejects paths and links that leave the prepared tree. Internal
relative library/framework symlinks remain supported.

The shared core is in `packaging/updater/core.cpp`; platform UI and handoff code
are alongside it. Native tests exercise replacement, font merging, interrupted
installation/rollback recovery, integrity failures, and cleanup. See the
[build guide](../packaging/build.md) for compiler and platform requirements.

## Release workflow

The build workflow uploads a checksum manifest for every variant alongside the
archive or all of its numbered parts. The release workflow:

1. Requires a successful `Build Executables` run from this repository and verifies
   that its commit is the selected tag's commit.
2. Downloads all platform artifacts and their build-time checksum manifests.
3. Checks file coverage, hashes, contiguous part numbering, complete joined hashes,
   archive version, platform, and format. Missing Windows, Linux, or macOS outputs
   fail the release.
4. Uses the matching `CHANGELOG.md` version entry for **What's new**, followed by
   download instructions, and generates `latest.json` and the combined
   `SHA256SUMS.txt`. If there is no matching entry, **What's new** is omitted.
   GitHub's generated **What's Changed**, **New Contributors**, and
   **Contributors** sections are not added.
5. Uploads the assets to a draft, with `latest.json` last, and checks the complete
   remote inventory, sizes, and GitHub's SHA256 digests.
6. Leaves the verified release as a draft and links to it in the workflow summary.
   CI never publishes it automatically. After **Release** succeeds, review the
   draft on GitHub and click **Publish release** to approve publication.

The workflow marks non-stable version tags as prereleases so they do not update
the stable release pointer when published. Keep that designation when reviewing
the draft. Reruns can resume a draft; published releases cannot be overwritten.

Version tags must agree with the version embedded in archive names. Use a new tag
and version to replace a published release. A failed upload leaves a draft that
the updater cannot discover. The manifest describes archive bytes, not user
configuration, fonts, downloaded model weights, or caches.

To recover the failed **0.3.0** release, open **Actions → Release → Run workflow**,
select the branch containing the release fix, and enter `run_id: 42878638201` and
`tag: 0.3.0`. This resumes the draft using the existing successful build, provided
its artifacts have not expired; no rebuild or tag change is needed. Rerunning the
original failed workflow uses its old definition, so start a new manual run.
Wait for it to succeed before reviewing and publishing the draft.

The stable discovery URL is:

`https://github.com/iamokish/foxreader/releases/latest/download/latest.json`

The updater accepts only HTTPS download URLs naming this repository and the
manifest's release tag, and checks redirects against GitHub's release/CDN hosts.
SHA256 verifies integrity; trust in the publisher comes from the GitHub repository
and HTTPS. Where present, GitHub build attestations provide additional provenance
that can be checked with `gh attestation verify`.

## Manifest schema, version 1

`platform` is singular. OS names are `windows`, `linux`, and `macos`; architecture
names are `x86_64` and `arm64`. Each architecture contains a `variants` map so CPU,
CUDA, and builds with or without GGUF remain separate. The example below omits
other variants and abbreviates hashes:

```json
{
  "schema_version": 1,
  "version": "0.2.2",
  "tag": "v0.2.2",
  "notes": "Release notes",
  "pub_date": "2026-09-22T00:44:18.046Z",
  "release_url": "https://github.com/iamokish/foxreader/releases/tag/v0.2.2",
  "platform": {
    "linux": {
      "x86_64": {
        "variants": {
          "cpu-gguf": {
            "device": "cpu",
            "gguf": "cpu",
            "archive": {
              "name": "fox-reader-v0.2.2-cpu-gguf-linux-x64.tar.gz",
              "format": "tar.gz",
              "size": 1500000000,
              "sha256": "64 lowercase hex characters for the complete joined archive"
            },
            "parts": [
              {
                "name": "fox-reader-v0.2.2-cpu-gguf-linux-x64.tar.gz.001",
                "url": "https://github.com/iamokish/foxreader/releases/download/v0.2.2/fox-reader-v0.2.2-cpu-gguf-linux-x64.tar.gz.001",
                "size": 1000000000,
                "sha256": "64 lowercase hex characters for this part"
              },
              {
                "name": "fox-reader-v0.2.2-cpu-gguf-linux-x64.tar.gz.002",
                "url": "https://github.com/iamokish/foxreader/releases/download/v0.2.2/fox-reader-v0.2.2-cpu-gguf-linux-x64.tar.gz.002",
                "size": 500000000,
                "sha256": "64 lowercase hex characters for this part"
              }
            ]
          }
        }
      }
    }
  }
}
```

`gguf` is `null` when omitted from the build, otherwise its backend device (such
as `cpu`, `cu130`, or `macos`). An unsplit archive still has one entry in `parts`,
with the archive's own name, size, and hash. A split archive's `archive` describes
the joined file; only its numbered parts are attached to the release.

For local release validation, with downloaded build checksums in separate
subdirectories:

```text
python packaging/release_manifest.py generate --artifacts artifacts --checksums checksums --repository iamokish/foxreader --tag v0.2.2 --notes-file release_body.md --output artifacts/latest.json --require-platform windows --require-platform linux --require-platform macos
```
