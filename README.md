# Fox Reader

Manga reader with bubble detection, OCR, translation, and typesetting.

Open a folder of pages, capture speech bubbles (rectangle, freehand, auto-detect, or whole page), OCR them, translate them, style each entry, then render clean typeset pages to a destination folder — all locally, with optional DeepL / JPDB / custom endpoints.

- Local-first: bubble detection, PaddleOCR, local MTL models (Gemma, VNTL), OpenCV / TextSeg text cleaning.
- Live SVG overlay preview that mirrors the backend renderer (wrapping, spacing, scale, tilt, spin, shift).
- Per-entry layers, backgrounds, fonts, outline, spacing, transforms, speaker tags, and detector tuning.
- Standalone executables built with Nuitka (no install needed on the target machine).

*This is an early release. Expect bugs, rough edges, and ongoing changes as Fox Reader continues to develop.*

**Version:** `0.1.0` (`pyproject.toml`). The C launcher reports its own `FOX_LAUNCHER_VERSION "1.0.0"` (`packaging/launcher/launcher.c`).

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [How to Use](#how-to-use)
- [Per-Entry Options Reference](#per-entry-options-reference)
- [Text Clean & Typeset Reference](#text-clean--typeset-reference)
- [Translation Providers](#translation-providers)
- [Custom Endpoints (User Translation APIs)](#custom-endpoints-user-translation-apis)
- [Hotkeys](#hotkeys)
- [Settings & Configuration](#settings--configuration)
- [Environment Variables](#environment-variables)
- [Installation in Depth](#installation-in-depth)
- [Building Executables](#building-executables)
- [Download](#download)
- [Releases](#releases)
- [Development](#development)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Credits & Third-Party](#credits--third-party)

## Features

### Read & navigate

- Image gallery with lazy thumbnails, keyboard navigation, and saved-page badges.
- Work mode (edit) and Reader mode (read) with fit-to-height / fit-to-width, zoom, pan, fullscreen, and webtoon strip mode.
- `Original / Saved` switch per page: compare the source scan against the typeset copy in the destination folder.
- Displayed at `http://127.0.0.1:7954` by default (`config/fox_config.yaml`).

### Capture & OCR

- Four capture flows: rectangle (`c`), freehand (`f`), bubble auto-detect, whole-page capture.
- Right-to-left reading order by default, switchable to left-to-right, with manual reorder + resort.
- Two OCR engines, switchable in Settings: classic PaddleOCR and PaddleOCR-VL (GGUF vision-language side job).
- Greyscale toggle, Auto-OCR-on-select, click-to-re-read per entry.

### Translate

- Built-in buttons per language: DeepL (`/translate/deepl`, JA/ZH/KO), JPDB (`/translate/jpdb`), local MTL (`/translate/ml`), plus user-defined custom endpoints (`/translate/custom`).
- Local MTL models: Gemma Q6/Q8 (JA/KO/ZH, GGUF via llama.cpp), VNTL Llama-3 8B (JA-EN with translation context + character roster).
- Auto-translate switch, translation context (previous pairs for context-aware models), character/speaker metadata for VNTL models, custom endpoint editor with preview/test/activate.

### Entries & typesetting

- Entry list = reading order (cards numbered 1..n, same numbers stamped on the overlay).
- Per-entry layer stack 1 (bottom) – 10 (top), gap-free and stable-sorted like the backend composite.
- Backgrounds: `auto` (region dominant colour), `color`, `transparent`, `clean` (repaint artwork via text-clean).
- Fonts served with data URIs + live preview; per-glyph symbol fallback (NotoSansSymbols2) at render time.
- Geometry: word/line spacing (fit-time), X/Y font scale (fit-time), X/Y shift in image pixels + X/Y/Z rotation in degrees (post-fit, text-only — the plate never moves).
- Inline edit-in-place over the bubble (`Enter` commits, `Shift+Enter` newline, `Esc` cancels).

### Text clean & render

- Clean methods: `region` (selected region), `ppocr` (PaddleOCR boxes), `textseg` (glyph UNet++). Reconstruction fills: `hybrid-level`, `hybrid`, `level`, `pyramid`, `telea`, `ns`.
- Detector tuning for TextSeg: speed presets (`best`/`fast`/`fastest`/`single`), tile sizes (`Off`/`256`/`512`/`1024`/`2048` with paired overlap), `glow` / `tta` / `transport` switches.
- Preview renders to `cache/` with progress polling (`/api/progress/{name}`), then Save promotes it (warns `409 exists` / `stale_preview` instead of overwriting blindly).
- Live overlay preview (toggleable) mirrors `src/fox_reader/typeset.py`: same wrap/hyphen rules, same binary-search font fit, explicit per-word spacing, CSS-3D perspective tilt with `cos` fallback.

### Productivity

- Split flow: draw a cut across a region to divide it; pieces inherit styling and slot.
- Characters roster (VNTL speaker tags) with CSV import/export.
- Workspaces, layouts (`basic` / `default`), dark/light themes.
- Session + folder state APIs, single-instance launcher, graceful shutdown (`POST /api/shutdown`).

## Quick Start

### Prerequisites

- Python `>=3.12,<3.13`
- [`uv`](https://docs.astral.sh/uv/) installed
- For the UI build: Node.js `22+` and `pnpm` (`pnpm@11`, CI uses Node 24)

### Install & run (source)

Pick the PyTorch backend matching your system. `<backend>` is one of: `cpu`, `macos`, `cu118`, `cu126`, `cu129`, `cu129_win`, `cu130`, `rocm71`, `rocm72` (ROCm wheels are Linux-only).

```bash
uv sync --extra dev --extra cpu
uv run --extra cpu fox-reader
```

Or with the shortcuts (CPU backend):

```bash
# Windows
run.bat

# Linux / macOS
./run.sh
```

Then open `http://127.0.0.1:7954`. First launch redirects to `/setup`, which downloads the required models (bubble detector, OCR, MTL) from Hugging Face and marks each with a `_completed` file.

### GGUF support (local GGUF MTL)

`llama-cpp-python` ships no wheels — every install compiles llama.cpp, and the `CMAKE_ARGS` decide whether the result has a GPU backend. It is therefore installed on its own via `packaging/llamacpp.py`, never through `uv sync --extra gguf`:

```bash
uv run --extra <backend> python packaging/llamacpp.py --device <backend> --install
```

`<backend>` mirrors the torch extras (`cpu`, `cu129`, `rocm72`, `macos`). On CPU it looks for OpenBLAS (offers a download into `nc/`); on CUDA/ROCm it needs the matching toolkit (`nvcc` / ROCm 6.1+). Preview without installing, or install by hand (note `--no-cache-dir` is mandatory — uv's cache key ignores `CMAKE_ARGS`):

```bash
python packaging/llamacpp.py --device cu129 --print-args
CMAKE_ARGS="<printed line>" uv pip install --no-cache-dir llama-cpp-python
```

Windows notes: hand installs need `CMAKE_GENERATOR` too (`MinGW Makefiles` + MinGW compilers, or Ninja inside a VS Developer Prompt for MSVC). MinGW builds also need `-D_WIN32_WINNT=0x0A00` (printed as `CFLAGS`/`CXXFLAGS`) and copy `libgcc`/`libstdc++`/`libwinpthread` DLLs next to the built lib — `--install` handles all of this automatically.

Useful flags: `--openblas {auto,download,off}`, `--cpu-baseline {avx2,avx,sse42}`, `--cuda-arch` / `--hip-arch`, `--python PATH`, `--find-openblas`.

## How to Use

1. **Load a folder.** Type a source folder path (or pick it in the folder modal) and optionally a destination folder. Loading sets the gallery, page dimensions, and which pages already have a typeset copy (`saved` badges). When source and destination are the same folder, saving overwrites and the `Original / Saved` switch hides.
2. **Pick a page.** Click a gallery thumbnail or use `ArrowRight` / `ArrowLeft` (`h` toggles the gallery). The viewer fits the page; zoom with `+` / `-` / `0`, pan by dragging.
3. **Capture regions.** Press `c` (rectangle), `f` (freehand), the bubble button (bubble auto-detect), or page capture (whole-page OCR+translate). Drag on the page; `Esc` cancels. Each capture becomes an entry card (reading order = list order).
4. **OCR.** Click an entry (or its OCR button) to read the region into the editor. `Auto OCR` reads entries with no text on select; greyscale and language (`languageSelect`: japanese/chinese/korean) apply.
5. **Translate.** With `Auto TL` on, translation follows OCR automatically. Otherwise click the entry's Translate button (DeepL / JPDB / MTL / custom, per language). Results land in the translation panel (`#translatedText`).
6. **Confirm.** `Confirm Entry` stores editor + panel text onto the captured region. Re-confirming the same region replaces text but keeps styling.
7. **Style entries.** Expand an entry (`expand_more`) for fonts, colours, layer, alignment, speaker, background, clean settings, spacing/scale/shift/rotate (see reference below). Every control repaints the overlay immediately (`refresh`); structural clean changes rebuild the panel (`rebuild`).
8. **Preview & save.** `Preview` renders the current page to `cache/` in a modal (progress polled); `Save` writes it to the destination folder (`Save` in the modal commits the preview token). `Generate` renders straight to the destination. Existing files need overwrite consent (`409`).
9. **Compare.** Flip `Original / Saved` on pages with a saved copy. Editing locks while viewing Saved (entries describe the original).

Tips: right-click a card or region to hide it from the render; layers decide paint order (10 on top); empty text + `clean`/colour background erases a bubble; `transparent` draws text with no plate.

## Per-Entry Options Reference

| Group | Controls | Values |
|---|---|---|
| Layer / Align / Speaker | Layer stepper, alignment segmented, speaker select | Layer `1–10` (ceiling = highest other + 1); align `center/right/left`; speaker `None` + character roster (`meta_id`) |
| Font / Outline | Family select, size select, font colour; outline width + colour | Size `Auto` + `6–72`; outline `Auto` + `0–32` (auto ≈ `round(size×0.09)`, min 1); colours `Auto` (detected) or hex |
| Spacing / Scale | `W` word spacing, `L` line spacing, `X`/`Y` stretch | `W` `Auto` + `0.0–3.0×`, `L` `Auto` + `0.5–3.0×` (tenths); scale `0.5–2.0×` (neutral `1.0`, part of the font fit) |
| Shift / Rotate | `X`/`Y` nudge (image px), `X`/`Y`/`Z` rotation (deg) | Shift bounded by page when loaded, backend clamps to ink; angles `−180–180` (X/Y tilt plane, Z spins clockwise, all about block centre, text-only) |
| BG | Mode select + colour | `auto` / `color` (+hex) / `transparent` / `clean` |
| Clean (when `BG=clean`) | Method + fill + tweaks | Method `region`/`ppocr`/`textseg`; fill `hybrid-level/hybrid/level/pyramid/telea/ns`; speed `best/fast/fastest/single`; tile `Off/256/512/1024/2048`; `glow`/`tta`/`transport` per method knobs |

Defaults for untouched entries reproduce the pre-geometry render exactly (`word=None, line=None, scale=1, shift=0, angle=0`).

## Text Clean & Typeset Reference

- **What runs:** clean-mode entries repaint artwork first (detection shared across frames), then entries composite bottom-layer-first (`typeset._ordered`), each with its plate + fitted text + post-fit transforms.
- **Typeset fit:** largest `6–72` px that fits the padded type area (12% horizontal / 10% vertical padding), binary-searched then verified; spacing + scale participate in the fit, shift/rotation apply afterwards to the lettering only.

### Detection — Method (where the lettering is)

One detection pass runs per page, not per entry: entries sharing a detector (and, for TextSeg, the same tuning) share one result over the union of their regions. Animated pages detect once and reuse it across frames.

| Method | What it removes | Cost | Knobs |
|---|---|---|---|
| `region` (default) | Your selection verbatim — everything inside the shape, no model, cannot miss | Instant | None (`fill` + `transport` still apply) |
| `ppocr` | PP-OCRv6 detector quads, one per text line (~2 s/page, model already loaded) | Fast | None — box coverage has no halo for Glow to absorb (`fill` + `transport` still apply) |
| `textseg` | Glyph-level UNet++ segmentation + stroke-radius-relative mask refinement (counters filled, specks dropped, glow absorbed) | Slowest, most precise (20–40 s/page on CPU; runs on the padded union crop, not the full page) | `glow`, `speed`, `tta`, `tile` |

Preview marking: `region` = blurred plate, `ppocr` = diagonal hatch, `textseg` = dots. Unknown method names from stale projects fall back to `region` rather than failing the page.

### Reconstruction — Fill (how the pixels underneath are rebuilt)

Each text cluster is rebuilt inside its own neighbourhood (local crop, grown until ≥45% known pixels), largest first, so nearby paint — not distant art — is the source. Flat-paper holes are pre-filled with the ring median (avoids grey ghosts on white bubbles). Ordered best-first in `INPAINT_METHODS`; the default is the first non-`transport` entry because `transport` costs seconds per cluster. What the panel offers depends on the OpenCV build (`caps` probes `xphoto.inpaint` + flag, since OpenCV 5 ships an empty `xphoto` module that imports but cannot run): pure-xphoto fills hide when unavailable (degrading would leave Telea wearing their name), while hybrids degrade their detail term to Telea and keep their pyramid shading intact. Unknown fills fall back to the default.

| Fill | How it works | Best for / watch out |
|---|---|---|
| `hybrid-level` (default) | Pyramid shading + FSR high-frequency detail, level-fixed | Painted artwork; ~1 s/cluster trade that holds up |
| `hybrid` | Pyramid shading + transported edge detail (Telea when xphoto missing) | Soft art needing edge continuation |
| `hybrid-fsr` | Pyramid shading + FSR detail variant | Line art / hatching over shading |
| `hybrid-patch` | Pyramid shading + patch-synthesis detail (real paint texture) | Textured paint where grain matters |
| `level` | Pyramid solver with boundary-implied re-levelling | Smooth gradients with tone steps |
| `pyramid` | Push-pull interpolation + matched grain (no invention) | Soft airbrushed fills; hazes on very wide holes |
| `patch` | Multiscale patch synthesis | Real texture; can mismatch shading / ridge on wide holes |
| `transport` | Offset (shiftmap) cover with whole displaced pieces + tone fix | Structure at every scale (hair, folds); can invent content; ~20 s |
| `fsr` / `fsr-fast` | Frequency-selective reconstruction (BEST / FAST) | Straight edges continuing as themselves; hidden without xphoto flags |
| `shiftmap` | xphoto shiftmap in Lab | Offset fill; hidden without its flag |
| `telea` / `ns` | OpenCV diffusion inpaint (radius 7) | Always available; Telea is the universal fallback |

### Parameters (entry clean panel)

| Parameter | Applies to | Meaning |
|---|---|---|
| `method` | All | Detection source above (`region`/`ppocr`/`textseg`) |
| `fill` | All | Reconstruction above (`hybrid-level` default) |
| `glow` (default on) | `textseg` only | Absorb the feathered glow around text into the mask (`glow_k 2.0` vs `0.0` = CLI `--no-glow`); distances in stroke radii so cover titles and furigana behave |
| `transport` (default on) | All fills | Allow escalation: wide holes (`half-width ≥ 60 px`) in textured paint (median high-freq `≥ 2.0` in the 2–10 px shell) escalate smear-prone fills to offset `transport`; wide holes in flat paint correctly stay smooth. Needs xphoto `transport` or escalation stays off. Reconstruction groups by `(fill, transport)` |
| `speed` (default `fastest`) | `textseg` only | `best`: flips + scales `(1.0, 0.6, 0.4)` + inverted pass (≤24 passes) · `fast`: no flips (≤6) · `fastest`: no inverted pass either (≤3; within 6.2% of `best` for ⅛ the cost — the interactive default) · `single`: one scale (1 pass, cheapest, no cross-scale vote to filter artwork). Inverted pass auto-skips on pages with no dark ground. Stale names → `fastest` |
| `tta` (default on) | `textseg` only | Flip test-time averaging (4 views). Only `best` ships flips, so only there does off save 4×; elsewhere the switch is stored but inert (dimmed, still clickable) |
| `tile` (default `Off`) | `textseg` only | `Off` (whole image, only whole-page view) / `256+48` / `512+96` / `1024/2048+192` overlap, cosine-blended. Small tiles exist for machines that cannot hold page activations, not for better detection. Only tabulated sizes accepted |

Switching method never loses tuning: `speed`/`tta`/`tile` ride along so toggling back restores them.

## Translation Providers

| Provider | Endpoint | Notes |
|---|---|---|
| DeepL | `POST /translate/deepl` | Needs API token in Settings; buttons hidden until `/api/translate/status` allows; `:fx` keys use the free host |
| JPDB | `POST /translate/jpdb` | Japanese parser API, token in Settings |
| Local MTL | `POST /translate/ml` | Gemma Q6/Q8 (JA/KO/ZH, GGUF); VNTL Llama-3 8B (JA-EN, context + characters). Load/unload + VRAM check via `/ml/*` |
| Custom | `POST /translate/custom` | User-templated endpoints (editor at `/user_endpoints`, preview/test/activate per language) |

Context pairs and character links ride only on MTL/custom requests, gated by the Context/Characters switches.

## Custom Endpoints (User Translation APIs)

Any HTTP translation API can become a translator button. Open the editor at `/user_endpoints` (linked from `/settings` as “Manage endpoints →”). Each language holds up to `20` endpoints, of which `3` can be active at once — active endpoints become translator buttons in the reader next to DeepL/JPDB/MTL. Changes save immediately; endpoints persist in `config/user_endpoints.yaml`.

### Lifecycle: from API docs to a button

1. **Add Endpoint** — starts a blank draft (default request `{text, src_lang, target_lang}`, default response `{text, error}`).
2. **Fill the blocks** below (connection → wire format → languages → request schema → response schema → query/headers → encryption).
3. **Preview request** — `POST /api/user_endpoints/preview`. No network: the server validates the draft and returns exactly what would be sent (`method/url/headers/body`) plus schema examples. List fields are filled with samples (`[[こんにちは, Hello], …]`, one roster entry + links) so you see their shape. When editing a saved endpoint, the form has no copy of the stored key, so the preview borrows it rather than failing.
4. **Save Endpoint** — `POST` (new) / `PUT` (edit). Validation errors come back as `error` with dotted paths (e.g. `request_schema.children.text: …`). The key is write-only and stripped from every browser-visible response.
5. **Test** — per-endpoint Test sends one **real** request with `こんにちは` (or your text), empty history, and shows `sent / translated / alternatives / message` plus the exact request. Use it before activating.
6. **Activate per language** — `PUT /api/user_endpoints/{id}/active` with `{"language": "japanese"}` (DELETE deactivates). The reader fetches `GET /api/user_endpoints/active` and renders one button per active language.
7. **Translate** — in the reader, select text and click your endpoint’s button (or Auto TL if it routes there). History and roster ride only when the schema carries those nodes (and the Context/Characters switches are on). Selections over `max_text_length` are refused client-side (`413`) before anything is billed or rate-limited.
8. **Manage** — edit, delete (`DELETE /api/user_endpoints/{id}`), or swap actives at any time.

### Connection block

| Field | Meaning |
|---|---|
| Name | Button label — max `6` chars, sits beside the built-in engines |
| Method | `GET/POST/PUT/PATCH/DELETE` |
| Hostname / URL | Host with optional path; a pasted `https://` prefix is fine. `<TEXT>`, `<SOURCE_LANG>`, `<TARGET_LANG>` are substituted — e.g. `api.example.com/<SOURCE_LANG>/translate` |
| Scheme / Port / HTTP | `http/https`, optional port `1–65535`, HTTP version `Default/2` (HTTP/2 via httpx) |
| Timeout | Seconds per attempt (`> 0`, default `20`) |
| Retries | `0–5`. Retries timeouts and `408/425/429/500/502/503/504` with exponential backoff (`0.4s → 4s` cap, honours `Retry-After`) |
| Max text length | `100–10000` chars (default `2000`). Longer selections fail before sending |
| Target language | Sent wherever the schema uses Target Language (default `en`) |

### Wire format block

- **Request body** (`body_format`): `json` (JSON body), `form` (urlencoded body), `query` (payload appended to URL, no body), `text` (raw body). `Auto` = `query` for `GET/DELETE`, else `json`. `form`/`query` need a flat object at the request root.
- **Response body** (`response_format`): `json` (decoded + read through the response schema) or `text` (the whole body **is** the translation; the response schema is then unused).
- **Repeat list keys (doseq)**: `false` sends a list as one joined value; `true` repeats the key (`dt=t&dt=at`). Applies to `form`/`query` payloads.
- **Join response chunks with** (`text_join`): separator when a repeat template yields several fragments (`\n`, `\t` understood, else `""`).
- **Static query parameters**: always appended to the URL (placeholders substituted); a `query`-format body overrides same-named keys.
- **Request headers**: sent as entered (placeholders substituted); an explicit `Content-Type` overrides the body-format default (`application/json`, `application/x-www-form-urlencoded`, `text/plain; charset=utf-8`).

### Languages block

Tick the languages this endpoint serves (editor suggests `japanese/korean/chinese`; the list itself is free-form lowercase). The optional per-language **code** is what the API expects (`japanese → ja`); it is sent wherever the schema uses Source Language. One translator button is created per active language.

### Request schema tree

Typed JSON nodes: `dict` (named `children`), `list` (`items` and/or response-only `each`), and primitives `string/int/float/bool/null` plus `uuid` (request-only; mints a fresh id per request — variant `uuid1/uuid2/uuid4`, prefer `uuid4` since `uuid1` embeds this machine’s MAC).

Dynamic **sources** (the only moving parts; everything `static` is sent verbatim):

| Source | Sends |
|---|---|
| `text` | The selection (exactly one required) |
| `src_lang` / `target_lang` | Language codes (at most one each, optional) |
| `context` | Earlier `[[source, english], …]` pairs, oldest first (at most one; only when the node exists — otherwise requests are byte-identical with or without history) |
| `character_info` + `context_character_links` | Roster `[{meta_id, name_en, name_ja, gender, alias_en, alias_ja}, …]` + one speaker id (or `null`) per context pair. Always **both or neither** |
| `error` / `alt` | Response-only — rejected in a request schema |

Rules enforced on save: exactly one `Text`; repeat templates (`each`) and `uuid` only in requests where documented; `Text`/`src_lang`/`target_lang` must be `string` nodes; list-array sources must be bare `list` nodes (they consume the whole array, no `items`/`each` inside).

**Placeholders** work inside any static string, header, query value, or the hostname itself:

| Token | Expands to |
|---|---|
| `<TEXT>` | The selection (sealed first when request encryption is on — even in the URL, which is the most-logged part) |
| `<SOURCE_LANG>` | Source code for the ticked language |
| `<TARGET_LANG>` | Target language field |

### Response schema tree

- Exactly one **Translated Text** node (any shape — string, chunk list, or `[text, meta]` pairs are flattened and joined with `text_join`).
- Optional **Error** node (string/null): an API error message beats a bare HTTP status.
- Optional **Alternatives** node (string/list): extra candidates, deduplicated, minus the main text.
- **Repeat template** (`each`) on a list collects an unknown number of chunks (e.g. per-sentence arrays) into one translation.
- With `response_format: text`, skip the tree: non-2xx returns `HTTP <code> — <snippet>`, empty bodies and undecryptable replies are `502`s, never silent.

### Encryption block (optional, Fernet)

Off unless a key is given. When on, the selection is sealed before it leaves this machine and the reply is opened on arrival — proxies, middleboxes, and request logs carry opaque tokens instead of the text. The endpoint holds the same key and reverses it: this hides content **from the network, not from the API itself**. Token length still tracks plaintext length, and timing/count are visible.

- **Key types**: `fernet` (paste the 44-char `Fernet.generate_key()` string) or `passphrase` (words stretched with scrypt, fixed salt `fox-reader/user-endpoint/v1`, `N=16384, R=8, P=1` — a compatible server must use these verbatim, then base64url the 32 bytes).
- **Directions are independent**: encrypt-before-sending and/or decrypt-on-arrival (at least one must be chosen). Names/aliases seal per-field (`meta_id`/`gender` stay plaintext so links still join); context pairs seal per side; response fragments open one by one, then join.
- **Rotation**: up to `4` keys, whitespace/comma-separated — first encrypts, all are tried for decrypt. Add new-first, migrate the server, drop the old.
- **Replay guard** (`max_age`, `30–86400`s, blank = off): sealed replies carry a timestamp; refuse stale ones only when both clocks are trusted.
- Requires the `cryptography` package; the editor disables the switch without it. The key never reaches the browser except for encryption (edit previews borrow the stored secret server-side).

### Copy-paste starters

**1. Minimal JSON POST** (`https://api.example.com/v2/translate`):

```json
// request root (dict): {"text": Text, "source": Source Language, "target": Target Language}
{"text": {"type": "string", "source": "text"},
 "source": {"type": "string", "source": "src_lang"},
 "target": {"type": "string", "source": "target_lang"}}
// response root (dict): {"translatedText": Translated Text, "message": Error}
{"translatedText": {"type": "string", "source": "text"},
 "message": {"type": "string", "source": "error"}}
```

**2. GET query API** (method `GET`, body `query`): same request tree; add static query `{"key": "<YOUR_KEY>"}` and a header if needed. The payload rides in the URL; over-long selections hit `max_text_length` first.

**3. Chunked reply with alternatives**: response root `dict` → `data: dict` → `sentences: list(each: string→Translated Text)` plus `candidates: list→Alternatives`, `err: string→Error`, `text_join: "\n"`.

### Limits & errors at a glance

- `20` endpoints / language, `3` active / language, name ≤ `6` chars, `max_text_length 100–10000`, `retries 0–5`, `timeout > 0`, `max_age 30–86400` or blank.
- `400` empty text · `404` unknown endpoint · `409` never (endpoints overwrite by id) · `413` selection too long · `500` unbuildable request · `502` bad JSON / missing Translated Text / undecryptable · `503/504` unreachable / timeout · `429/5xx` retried per policy.
- Automatable: `GET /api/user_endpoints`, `GET …/active`, `POST …/preview`, `POST /`, `PUT /{id}`, `DELETE /{id}`, `POST /{id}/test`, `PUT|DELETE /{id}/active`.

## Hotkeys

Typing in inputs/textareas (or the editable translation panel) disables hotkeys. Loading overlay, backend-status overlay, and folder modal also suppress them.

**Work mode** (default):

| Key | Action |
|---|---|
| `c` | Rectangle capture |
| `f` | Freehand capture |
| `h` | Toggle gallery panel (re-fits viewer) |
| `ArrowRight` / `ArrowLeft` | Next / previous page |

**Reader mode** (`workspace-reader`):

| Key | Action |
|---|---|
| `ArrowRight` / `ArrowDown` / `Space` | Next page |
| `ArrowLeft` / `ArrowUp` | Previous page |
| `f` | Fullscreen toggle |
| `s` | Strip (webtoon) mode toggle |
| `+` / `=` , `-`, `0` | Zoom in / out / refit |
| `Esc` | Back to work mode |

**Editors & dialogs:**

| Context | Key | Action |
|---|---|---|
| Inline bubble editor | `Enter` / `Shift+Enter` / `Esc` | Commit / newline / cancel |
| Colour hex box | `Enter` | Commit parsed hex |
| Eyedropper / capture stroke | `Esc` / right-click | Cancel pick / cancel stroke |
| Preview modal, character/appearance modals | `Esc` | Close |
| Folder path input | `Enter` | Load folder |

## Settings & Configuration

On-disk state (repo root, gitignored except `fox_config.yaml`):

- `config/fox_config.yaml` — `host: 127.0.0.1`, `port: 7954`, `mtl_dir: models`, `theme: dark|light`, `layout: basic|default`. `host`/`port` are immutable at runtime.
- `config/settings.yaml` — `mtl_defaults` per language (`japanese/korean/chinese`), `devices` (`paddleocr/paddleocr_vl/bubble/translator/textseg: auto|<device>`, `mtl_threads`), `deepl_api_token` / `jpdb_api_token`, `ocr_engine: paddleocr|paddleocr-vl`.
- `config/user_endpoints.yaml` — custom translation endpoints.
- `config/workspaces.yaml` — panel layout sizes.
- `models/` — weights (bubble `model.safetensors`, PaddleOCR det/rec + VL GGUF, manga seg safetensors, MTL Gemma/VNTL). Downloaded once via `/setup`; each completed download touches a `_completed` marker.
- `fonts/` — your `.ttf`/`.otf` faces (scanned + cached in `.fonts_cache.json`); two fallbacks are compiled into the binary. `NotoSansSymbols2` is a render-time per-glyph fallback, never listed. Guide (adding faces, licensing, suggestions): [`/FONT.md`](/FONT.md).
- `cache/` — scratch previews/masks; cleared at startup and shutdown.

Pages:

- `/setup` — first-run wizard: required + MTL download state, per-file progress, optional OCR-VL side job, finish/close.
- `/settings` — devices (per-slot AUTO + thread count), per-language MTL defaults, API tokens (validated), OCR engine switch.
- `/user_endpoints` — custom endpoint CRUD, mustache-style preview, test call, per-language activation.
- Characters modal — VNTL speaker roster CRUD + CSV import/export; entry Speaker select tags `character_id`.

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `FOX_READER_LOG_LEVEL` | `INFO` | Log verbosity |
| `FOX_READER_ROOT` | exe dir / source root | Override project root (models/fonts/config resolution) |
| `FOX_READER_DISK_ASSETS` | off | Read embedded frontend/fonts from disk (dev) |
| `FOX_READER_MTL_THREADS` | settings heuristic | Override MTL thread count |
| `FOX_READER_IGNORE_MTL_MEMORY` | off | Skip the VRAM gate before loading local MTL |
| `FOX_READER_MTL_VERBOSE` | off | Verbose local-MTL (llama.cpp) logging |
| `FOX_READER_VL_VERBOSE` | off | Verbose PaddleOCR-VL logging |
| `FOX_READER_SHUTDOWN_TOKEN` | open | Gate `POST /api/shutdown` (launcher sets it) |

`USERNAME`/`USER` fall back to `fox-reader` for endpoint rendering.

## Installation in Depth

| Backend extra | Torch index | Platform |
|---|---|---|
| `cpu` | CPU | Win/Linux/macOS |
| `macos` | PyPI | Apple Silicon |
| `cu118` / `cu126` / `cu129` / `cu130` | matching CUDA index | Linux (CUDA toolkit needed for GGUF builds) |
| `cu129_win` | CUDA index (Windows split) | Windows |
| `rocm71` / `rocm72` | ROCm index | Linux only (no ROCm wheels on Windows/macOS) |

```bash
uv sync --extra dev --extra <backend>
uv run --extra <backend> fox-reader
```

Frontend (TypeScript + Vite + Preact):

```bash
cd frontend
pnpm install
pnpm dev      # watch mode (Terminal 1) + `uv run fox-reader` (Terminal 2)
pnpm build    # → static/app.js (gitignored IIFE bundle)
```

Windows `pnpm install` may fail with `UNKNOWN ... symlink`: enable Developer Mode (Settings → System → For developers) or `pnpm install --node-linker=hoisted`. `packaging/build.py` retries with that flag automatically.

## Building Executables

> New to this? Start with [`packaging/build.md`](packaging/build.md) — the
> copy-paste guide (what to download, which `--device` for your GPU, build,
> run) for Windows and Linux.

Standalone builds compile the backend with [Nuitka](https://nuitka.net/), embed the frontend, and ship a small C launcher (the user-visible entry point).

Result layout:

```
fox-reader/
  launcher(.exe)   starts backend, opens UI, closes both on exit
  bin/             compiled backend + linked libs (+ MSVC runtime DLLs + .dist-info metadata)
  cache/           scratch (empty, created at build)
  config/          fox_config.yaml
  fonts/           your fonts
  models/          weights (downloaded on first run)
```

### Prerequisites

- `uv`, `pnpm`, Node.js `22+` (CI pins 24 + pnpm 11), Python `3.12`
- C compiler for the launcher (verified by compiling+linking a smoke program; `python packaging/toolchain.py --which` shows the winner):
  - Windows: VS Build Tools (“Desktop development with C++”) or MinGW-w64 (`gcc` on `PATH`; use winlibs/MSYS2 `mingw-w64-x86_64-gcc`, not MinGW.org bare `gcc`). With no compiler, the build asks Nuitka to fetch its MinGW64.
  - Linux: `gcc`/`clang` (`apt install build-essential`); CUDA variants pin GCC per toolkit (11.8→11, 12.6/12.9→13, 13.0→14).
  - macOS: Xcode CLT (Metal builds only on macOS).
- GPU builds: matching CUDA toolkit (`nvcc`) or ROCm 6.1+; CPU GGUF builds want OpenBLAS (`apt install libopenblas-dev` on Linux, or `--openblas download`).

### Build commands

```bash
python packaging/build.py
python packaging/build.py --clean            # wipe previous artifacts first
python packaging/build.py --skip-frontend    # reuse prebuilt static/app.js
python packaging/build.py --keep-venv        # reuse build venv next time
python packaging/build.py --deep-compile     # compile deps too (hours)
python packaging/build.py --device cpu       # cpu|macos|cu118|cu126|cu129|cu129_win|cu130|rocm71|rocm72
python packaging/build.py --nuitka-arg ARG   # repeatable, forwarded to Nuitka
python packaging/build.py --no-blas          # alias for --openblas off
```

GGUF / CPU tuning:

```bash
python packaging/build.py --no-gguf
python packaging/build.py --gguf-device cpu          # CUDA torch + CPU llama.cpp
python packaging/build.py --openblas download
python packaging/build.py --cpu-baseline sse42       # avx2 (default) | avx | sse42
python packaging/build.py --cuda-arch 75-real,86-real,120-virtual
python packaging/build.py --hip-arch gfx1100,gfx1201
```

Defaults compile only `fox_reader` + entry (5–15 min); `--deep-compile` is the slow release path. Nothing targets the build machine specifically (`GGML_NATIVE=OFF`, pinned ISA, virtual GPU arches for JIT fallback). Archives are named `fox-reader-v{version}-{device}[-gguf[-ggufDevice]]-{platform}` (zip on Windows, tar.gz on Linux).

### Slimming the Linux tree

Shared libraries always travel as DLLs, never as data files: `--noinclude-data-files` covers `*.so*`/`*.dll`/`*.dylib`/`*.pyd`, because a versioned Linux library such as torch's `libgomp.so.1` slips through Nuitka's built-in filter, gets collected as data while the DLL scan claims the same path, and stops the build with a data-file-vs-DLL conflict.

Linux then still ships every shared library twice into `bin/`: once under its package (`torch/lib/libtorch_cpu.so`, where the package's own RUNPATH finds it — the layout Windows ships only, and works) and once flat at the top level (Nuitka's dependency scan, e.g. for torchvision). Both copies are live, so the build replaces the flat copies with relative symlinks instead of deleting them — every lookup path keeps resolving, the bytes ship once. Version triplets (`libllama.so`, `.so.0`, `.so.0.20.0`, symlinks in the wheel that arrive as three full copies) collapse the same way after a SHA-256 check. Also dropped, since nothing the app runs imports, spawns, or dlopens them: `torch/test`, `torch/include`, the test executables and protobuf compiler in `torch/bin` (`torch_shm_manager` stays), and `*test*` stubs in `torch/lib`. Unique transitive dependencies (e.g. hashed `libopenblas-*.so`) are never touched, ambiguous cases are kept with a warning, and any refused symlink keeps the original file. Set `FOX_BUILD_KEEP_DUP_LIBS=1` to ship the tree as Nuitka produced it.

### AMD on Windows

`--device rocm*` is refused on Windows (no ROCm torch wheels). AMD GPUs can still run GGUF models via llama.cpp HIP:

```bash
python packaging/build.py --device cpu --gguf-device rocm72
```

Needs the HIP SDK (`$HIP_PATH` / `hipconfig` / `%ProgramFiles%\AMD\ROCm`); the build drives ROCm `clang` + Ninja itself (MSVC cannot compile the device code).

## Download

Grab a ready-made build from
[GitHub releases](https://github.com/iamokish/foxreader/releases) when one
matches your machine — CPU builds (`...-cpu-gguf-...`) run anywhere, GPU
builds need that CUDA driver:

| You need | Get it |
|---|---|
| Windows or Linux, no NVIDIA GPU | Download the `cpu` archive for your OS |
| Windows + older NVIDIA GPU | Download a `cu118`/`cu126` archive if one is listed |
| Linux + NVIDIA GPU, RTX 40/50 series, or CUDA 12.9/13.0 | **Not provided — build it locally** (one command, see below) |
| AMD GPU | **Not provided yet** (ROCm builds are coming soon) — use `cpu` |

Missing your build? Make it yourself — no programming needed, just copy
and paste:

```bash
# Linux NVIDIA GPU, e.g. CUDA 12.9
python3 packaging/build.py --device cu129
```

Full walkthrough (what to install, which `--device` for your card, then
run): [`packaging/build.md`](packaging/build.md).

## Releases

- GitHub releases: `https://github.com/iamokish/foxreader/releases`
- Produced by `.github/workflows/build.yml` on `v*` tags (or manually): native matrix builds → one release with all active archives + auto notes.
- Active variants today: `cpu-gguf-linux`, `cpu-gguf-windows`, `cu118-gguf-windows`, `cu126-gguf-windows` (e.g. `fox-reader-v0.1.0-cpu-gguf-windows-x64.zip`). Further `cu129`/`cu130`/Linux-CUDA rows exist commented-out in the workflow — uncomment to enable.
- Pick the archive matching your OS + GPU (CPU builds run anywhere; CUDA builds need that CUDA driver; GGUF models run through the bundled llama.cpp with no toolchain needed).

## Development

```bash
cd frontend
pnpm lint / pnpm lint:fix
pnpm fmt / pnpm fmt:check
pnpm typecheck
```

Tests (default suite):

```bash
uv run --locked pytest tests/ -v
uv run pytest tests/test_backend.py -v
uv run pytest tests/test_frontend.py -v
uv run pytest tests/test_build.py -v
```

Gated/extended checks:

```bash
FOX_TEST_FULL=1 FOX_TEST_SLOW=1 FOX_TEST_DIST=1 uv run pytest tests/ -v
```

(`FOX_TEST_FULL` external tools, `FOX_TEST_SLOW` slow cases, `FOX_TEST_DIST` built-distribution checks: executable, DLLs, metadata, OpenCV, launcher single-instance, shutdown, assets/fonts, layout. Legacy `tests/manual/` scripts are non-strict `xfail` until updated.)

## Project Structure

```
foxreader/
  src/fox_reader/        FastAPI backend (routes/, services/, translate/, clean/, models/)
  frontend/              Preact + Vite UI (src/, templates/, static/)
    src/components/      Toolbar, Viewer, Gallery, Translation, Entries, FontSelector,
                         SVGOverlay, Folder, Appearance, Characters, common
    src/layouts/         Basic / Default (+ ActiveLayout switch)
    src/workspace/       work/reader modes, strip, overlays
    src/themes/          dark / light
    src/entries/         geometry, textFit (preview layout), shapes (overlay),
                         split, inpaint, pageTl
  packaging/             Nuitka build, toolchain + launcher (C), llama.cpp helper, asset embed
  config/                fox_config.yaml, settings.yaml, user_endpoints.yaml, workspaces.yaml
  fonts/ / models/ / cache/   faces / weights / scratch
  tests/                 backend + frontend + build suites (+ manual legacy scripts)
  .github/workflows/     build.yml (matrix + tagged releases)
```

## How It Works

```
Pages → [folder] → gallery
  → capture (rect/free/bubble/page) → entries (reading order)
  → OCR (PaddleOCR classic or VL) → editor text
  → translate (DeepL/JPDB/local MTL/custom) → translation panel
  → confirm → styled entries (layers, fonts, clean, geometry)
  → preview (/inpaint/preview → cache/) → save (/inpaint/save-preview or /generate → dest/)
```

- Backend (`src/fox_reader/app.py` lifespan): clears `cache/`, loads config/settings/endpoints/workspaces, gates local-MTL defaults, configures devices/threads, stands up session/font/progress singletons; serves embedded `static/` + `templates/` with no-cache (except `/font`).
- Bubble split rasterises the drawn stroke, rebuilds pieces from the mask, and hands them back as new regions.
- Clean groups jobs by detector/fill so detection runs once per page; typesetting (`typeset.py`) then composites layers bottom-up with per-entry plates, auto-fit sizes, symbol fallback, and text-only post-fit transforms.
- Frontend overlay (`entries/textFit.ts` + `entries/shapes.ts`) ports that layout (paragraphs, hyphen-only breaks, alignment, binary-search fit) and draws plate/text/clean-FX/outline/number badges in layer order.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `UNKNOWN ... symlink` on `pnpm install` (Windows) | Enable Developer Mode, or `pnpm install --node-linker=hoisted` |
| No C compiler for build | Install VS Build Tools (C++) or MinGW-w64 (winlibs/MSYS2); check `python packaging/toolchain.py --which` |
| CUDA build fails | Install exactly the toolkit matching `--device`; Linux CUDA pins GCC (see workflow); Windows CUDA drives MSVC |
| ROCm on Windows refused | Expected — use `--device cpu --gguf-device rocm72` + HIP SDK for GGUF |
| GGUF segfault after install | Reinstall with printed `CMAKE_ARGS` + `--no-cache-dir` (uv reuses wrong-device wheels otherwise) |
| Page shows setup wizard | Required models missing — complete `/setup` downloads (needs disk + network) |
| MTL will not load | Check `/ml/memory` and device pins in `/settings`; `FOX_READER_IGNORE_MTL_MEMORY=1` bypasses the gate for testing |
| Port in use | Change `port` in `config/fox_config.yaml` (host/port are file-only) |
| Stale UI after save | Flip `Original / Saved`; gallery badges mark pages with a dest copy |

## Credits & Third-Party

- OCR: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (classic + VL)
- Detection: PyTorch bubble segmentation (SafeTensors) + OpenCV (`opencv-contrib-python`)
- Local MT: Gemma / VNTL via [llama.cpp](https://github.com/ggerganov/llama.cpp) (`llama-cpp-python`)
- Online MT: [DeepL API](https://www.deepl.com/pro-api) (official v2 client), [JPDB](https://jpdb.io/) (official v1 client); custom user endpoints supported
- Packaging: [Nuitka](https://nuitka.net/), Vite, Preact, Hugging Face Hub, FastAPI/Uvicorn, Pydantic
- Fonts: ComicMono (bundled fallback) + NotoSansSymbols2 (render-time symbol fallback) — see `fonts/` licences
- Models download from Hugging Face on first run (`/setup`); GPU wheels from the PyTorch indexes per `--device`
