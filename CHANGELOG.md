# Changelog

Notable changes to Fox Reader are recorded here. Version entries are summarized
from the repository's Git tags. Unreleased changes have not yet been included
in a tagged version.

## [Unreleased]

### Added

- Native Yes/No close confirmation on Windows, Linux, and macOS. The dialog
  follows the system theme, stays above its parent window, and defaults to No.
- Backend health monitoring on setup, settings, and custom-endpoint pages,
  with themed Retry/Refresh notices above open dialogs. Restart recovery
  requires an explicit refresh, preserving unsaved forms until then.
- YAML copy/paste import and export for one custom endpoint at a time, with
  validation and correction suggestions. Imports open an editable draft before
  saving; shared encryption keys are excluded and must be entered separately.
- String roots in the custom-endpoint schema editor for bare text requests.

### Changed

- Custom-endpoint names must be unique across all endpoints and languages,
  ignoring case and surrounding whitespace. Existing duplicates remain
  available, but conflicting imports and edits require a new name before saving.
- Expanded the custom-endpoint guide with YAML sharing, validation rules,
  encryption-key handling, and management API examples.

### Fixed

- Custom-endpoint hostnames reject protocol prefixes and use the selected
  Scheme. Existing saved HTTP(S) URLs migrate while preserving their effective
  address, endpoint ID, activation, and encryption key.
- Health checks time out stalled requests, avoid overlapping polls, and keep
  restart notices visible until refresh. Escape and Tab no longer affect
  background dialogs while a health notice is open.

## [0.3.0]

### Added

- Native desktop windows alongside browser mode, with window recovery and
  update controls.
- Qwen3.5 4B and 9B local translation models, plus a smaller VNTL Llama-3 8B
  quantization.
- Per-model context-window settings, with 8K and 16K options where supported.
- Translation context and character/speaker metadata for all local translation
  models.
- Show/Hide controls for saved custom-endpoint encryption keys.
- GitHub issue forms, a pull request template, and this changelog.

### Changed

- Improved setup, model-storage selection, download progress, and reader
  controls.
- Native windows keep interface labels non-selectable while preserving text
  selection in editable fields and useful read-only output, including errors,
  license text, and endpoint request previews.

### Fixed

- Custom endpoints receive the current speaker ID and correctly aligned
  character and conversation-context metadata.
- Editing an endpoint preserves stored encryption keys and their encoding when
  unchanged; test previews show the actual request.
- Replaced native select and checkbox painting with shared styles to avoid
  WebKit theme mismatches and overflowing checkmarks.
- Prevented accidental text selection while dragging Vulpes panel splitters,
  with normal selection restored after release, cancellation, or focus loss.
- Reader/editor shortcuts no longer intercept macOS Command-key combinations
  such as copying selected text.
- Centered the close icon in the local-model memory warning.

## [0.2.2]

### Added

- Optional symbol fallback fonts, per-character font assignments, and automatic
  size/baseline calibration shared by preview and export.
- Apple Silicon builds for macOS 14 and newer, with Torch MPS and llama.cpp
  Metal support.
- CUDA 13.2 backend support.

### Fixed

- Font baseline alignment and typesetting differences between live preview and
  exported pages.
- Linux CUDA 11.8 builds.
- macOS architecture checks, Metal backend validation, and application signing
  during packaging.

## [0.2.1]

### Added

- Automatic text, outline, and background color detection from captured regions.
- Detector-assisted color measurement for PaddleOCR-VL.
- Split release archives, SHA-256 checksums, and build-provenance attestations.

### Fixed

- Color handling differences between clean/transparent overlays and exported
  pages.
- Vulpes status-bar layout.
- Windows line endings in checksum manifests.

## [0.2.0]

### Added

- Vulpes layout with a pages rail, canvas, editor, resizable sections, and footer
  actions.
- Shared reader views for single pages, double-page spreads, continuous
  scrolling, and original-versus-saved comparison.
- Adjustable padding around automatically detected speech bubbles.

### Changed

- Replaced the Default layout with Vulpes while retaining the Basic layout.
- Bundled the OCR support code needed by Fox Reader instead of depending on the
  full Transformers runtime.

## [0.1.0]

### Added

- Initial tagged Fox Reader version.
- Folder-based manga reading, region capture, bubble detection, and PaddleOCR
  and PaddleOCR-VL text recognition.
- Local Gemma and VNTL translation, DeepL and JPDB integration, and configurable
  translation endpoints.
- Per-entry typesetting, image cleaning, live previews, and export.
- Character rosters, custom fonts, dark/light themes, and standalone builds.

[Unreleased]: https://github.com/iamokish/foxreader/compare/0.3.0...master
[0.3.0]: https://github.com/iamokish/foxreader/compare/0.2.2...0.3.0
[0.2.2]: https://github.com/iamokish/foxreader/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/iamokish/foxreader/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/iamokish/foxreader/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/iamokish/foxreader/tree/0.1.0
