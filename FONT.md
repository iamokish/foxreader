# Fonts — how to add your own lettering

Fox Reader renders translations with the fonts in the `fonts/` folder.
It ships with a built-in fallback (ComicMono), so it works with the folder
empty — but manga looks far better with real comic lettering, which you add
yourself in under a minute.

## Adding fonts (all you do)

1. Get a `.ttf` or `.otf` file (only these two formats count — everything
   else in the folder is ignored).
2. Copy it into the `fonts/` folder:
   - **Built app:** the `fonts/` folder next to `launcher` / `launcher.exe`.
   - **Running from source:** the `fonts/` folder in the repo root.
3. Restart the app (or backend). New faces appear in the entry font picker
   automatically.
4. Pick a face per entry in the entry styling controls.

Notes:

- Bold/italic are separate files — copy each
  weight you want; faces group by their family name.
- The folder also holds `.fonts_cache.json` (the scan index — leave it
  alone, it rebuilds itself) and the fallback licences. A folder holding
  only those counts as empty, and the built-in ComicMono is used.
- `NotoSansSymbols2` is a hidden symbol fallback for missing glyphs; it is
  never listed as a pickable font.
- A broken font file is skipped with a warning, never a crash. If a face
  does not show up, the file is likely corrupt or not really TTF/OTF.

## Only use fonts you have the legal right to use

Font files are software with their own licences — separate from this app's
licence. **Do not copy a font into `fonts/` unless its licence allows your
use**, and never redistribute a paid font by sharing your `fonts/` folder.

Rules of thumb:

- **OFL (SIL Open Font License)** — the safest: free to use, share, and
  bundle, including commercially. Everything on Google Fonts is OFL.
- **Free-for-indie comic fonts** (e.g. Blambot, Comicraft freebies) — free
  for independent / small-press comics, but commercial or large-print use
  usually needs a paid licence. Read the licence that comes with the
  download; "free download" is not "free for everything".
- **DaFont and similar archives** — the licence differs per font (some are
  demo-only or personal-use-only). Open the font's own page/file and check
  before using it in anything you publish.
- **System / OS fonts** (the ones that came with Windows/macOS) are
  licensed with the OS — fine on your own screen, not necessarily fine to
  copy into `fonts/` on another machine or to embed in distributed work.
- **Scanlations and published pages count as publishing.** When in doubt,
  buy the licence or pick an OFL font.

## Suggestions

Proven manga-lettering pairings (all free in at least their base form —
re-check the licence at download time, terms change):

| Use | Fonts | Where to get them |
|---|---|---|
| Dialogue (the workhorse) | Anime Ace 2.0 BB, CC Wild Words Roman (+ Italic/Bold), Digital Strip 2.0 BB | [Blambot](https://www.blambot.com) (free indie pack), [Comicraft freebies](https://www.comicbookfonts.com) |
| Dialogue, guaranteed-free | Comic Neue, Patrick Hand, Kalam | [Google Fonts](https://fonts.google.com) (all OFL) |
| Narration boxes | Digital Strip 2.0 BB, Komika Text | Blambot / Comicraft (see above) |
| Big SFX | Badaboom BB, Blow Up BB, Komika Axis | Blambot (free indie pack) |
| Big SFX, guaranteed-free | Bangers, Luckiest Guy, Titan One, Bowlby One | Google Fonts (all OFL) |
| Handwritten notes, signs | Caveat, Gochi Hand, Short Stack, Shantell Sans | Google Fonts (all OFL) |

Start with three faces and you cover ~95% of pages: **Anime Ace** (dialogue),
**Digital Strip** (narration), **Bangers or Badaboom** (SFX). Fill gaps with
OFL faces from Google Fonts, which never need a second thought.
