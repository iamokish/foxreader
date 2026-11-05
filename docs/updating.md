# Updating Fox Reader

Open **Settings → Updates**, then select **Check for updates**. Checking is
explicit: opening Fox Reader does not contact GitHub. The app checks the latest
stable release and keeps the installed Windows/Linux/macOS, architecture,
CPU/CUDA, and GGUF choices. If the release lacks an exact match, use the release
page to choose a distribution yourself.

**Download update** saves the verified archive under `updates/<version>/` beside
the launcher. Every part is checked for its declared size and SHA256, and split
archives are joined in their numbered order and checked again. Completed parts
survive cancellation and failed downloads; a retry checks their hashes before
reusing them. The app never extracts, installs, or executes an update.

After the download finishes:

1. Close Fox Reader and extract the downloaded archive into a **new folder**.
2. Copy your `config`, `fonts`, and `models` folders from the old installation to
   the new installation. Keep the old folder as a backup.
3. Start the new launcher and check the setup and reader before removing the old
   installation. Keep the old version if the new one does not work for you.

Source checkouts and older releases without `build-info.json` can check release
notes, but download selection stays manual. A downloaded split archive needs
approximately twice its archive size in free space while joining; the app checks
space before transferring data and removes the numbered parts after a successful
join. It keeps your existing installation and user files throughout.

Local builds made with `--no-window` can check newer versions and release notes,
but cannot download an update archive through the app: published CI archives
include native windows. Rebuild the newer release with `--no-window` to keep Fox
Reader's graphical windows excluded.

## Release workflow

The build workflow uploads a checksum manifest for every variant alongside the
archive or all of its numbered parts. The release workflow:

1. Requires a successful `Build Executables` run from this repository and verifies
   that its commit is the selected tag's commit.
2. Downloads all platform artifacts and their build-time checksum manifests.
3. Checks file coverage, hashes, contiguous part numbering, complete joined hashes,
   archive version, platform, and format. Missing Windows, Linux, or macOS outputs
   fail the release.
4. Generates `latest.json` and the combined `SHA256SUMS.txt`.
5. Uploads the assets to a draft, with `latest.json` last. Before publishing, checks
   the complete remote inventory, sizes, and GitHub's SHA256 digests.
6. Publishes only after verification. Prereleases do not update the stable release
   pointer. Reruns can resume a draft; published releases cannot be overwritten.

Version tags must agree with the version embedded in archive names. Use a new tag
and version to replace a published release. A failed upload leaves a draft that
the updater cannot discover. The manifest describes archive bytes, not user
configuration, fonts, downloaded model weights, or caches.

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
