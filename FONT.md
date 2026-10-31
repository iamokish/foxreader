# Fonts

Fox Reader officially ships **ComicMono** (regular and bold) for text and
**NotoSansSymbols2** for missing symbols. Both are built in, so Fox Reader works
with an empty `fonts/` folder. Every other font is optional and supplied by you.

## Adding text fonts

1. Get a `.ttf` or `.otf` file. Other formats are ignored.
2. Copy it into the `fonts/` folder:
   - **Built app:** the folder next to `launcher` / `launcher.exe`.
   - **Running from source:** the folder in the repository root.
3. Restart Fox Reader or its backend. New faces appear in the entry font picker.
4. Pick a face in the entry styling controls.

Bold and italic faces are separate files; copy each weight you want. Faces
group by their family name. A broken font file is skipped with a warning. If a
face does not appear, check that the file is a usable TTF or OTF font.

The folder also holds `.fonts_cache.json` and font licences. Leave the cache
alone; it rebuilds itself. A folder holding only those files counts as empty,
and the built-in ComicMono faces are used. NotoSansSymbols2 is a hidden symbol
fallback and never appears as a pickable text font.

## Optional symbol fallback fonts

Fox Reader can use up to **three optional fonts you supply** to fill characters
missing from a text font. Add their files as above, then open **Settings → Fonts**.

1. Select fonts from the searchable list already loaded by Fox Reader. Names
   preview in their own face, with readable filenames underneath.
2. Use the up/down buttons to choose their priority, or remove a selection.
3. Optionally assign characters with **Paste characters → Add**, or open
   **Browse glyphs** to search by character, Unicode code (`U+2661`), or name
   (`heart`).
4. Click **Save Fonts**. Leave every slot and assignment empty for the default.

An assigned character always uses its assigned font, even if the text font or
an earlier fallback contains it. For every other character, Fox Reader tries
the selected text font first, then your first, second and third fallback in
order. **NotoSansSymbols2 always stays last** in the ordinary fallback
hierarchy, in the fixed fourth slot. Empty slots are skipped.

Each fallback, including NotoSansSymbols2, can own up to **25 distinct Unicode
characters**. A character can belong to only one font. Repeated characters in
one paste count once; supplementary characters such as `😀` also count once.
Strings assign each Unicode codepoint independently. Spaces in the paste field
are ignored; spaces and formatting controls cannot be assigned. Every
assignment must have a usable glyph in that font. The glyph browser disables
characters assigned to another face and pages its results. Click a character
chip or a selected glyph to remove its assignment.

Selected fallbacks disappear from text font pickers. Removing a fallback makes
it available for text again. Keep at least one font available for text. Saved
entries referencing a font subsequently made a fallback render with the first
available text font. Both live preview and backend export apply the same order
and calibration. Open readers in the same browser refresh after a save; reload
a reader in another browser to receive the change.

Reordering fonts preserves their assignments; removing a fallback also removes
its assignments. Clearing an assignment restores normal text-font and
missing-character fallback behavior.

Preferences are stored in `config/settings.yaml`. The optional filenames below
are placeholders for your own files:

```yaml
fonts:
  symbol_fallbacks:
    - MySymbols.ttf
    - OtherSymbols.otf
  symbol_characters:
    MySymbols.ttf: "♡★"
    OtherSymbols.otf: "■"
    NotoSansSymbols2-Regular.otf: "✓"
```

Additional fonts stay in your `fonts/` folder; Fox Reader does not embed or
distribute them. Missing saved fonts are skipped without erasing the
preference. If deleting fonts leaves every remaining font selected as a
fallback, Fox Reader reserves one for text until you adjust Settings.

## Automatic size and baseline calibration

Each fallback is calibrated independently against each text font. Calibration
uses actual glyph outlines in font units, so it does not depend on one raster
size, hinting, or a missing-character replacement box. It samples supported
capital letters for text height, or supported non-Latin letters when there are
no Latin capitals. For symbol height it uses supported common symbols, then a
bounded, evenly spread sample of symbols and private-use characters. Fonts
mapping artwork to ordinary letters are also supported as calibration inputs.
Empty outlines and duplicate glyph aliases are excluded. Median and median
absolute deviation filtering reduce the influence of tiny marks and oversized
logos.

The scale is bounded to 0.5–3 times the text size. Unmeasurable fonts use a safe
unscaled result. A font whose outlines are all empty cannot be selected as a
symbol fallback. Missing-character fallback retains each face's relative glyph
sizes and baseline.

For assigned characters that the text font also contains, calibration matches
the replacement's own outline height to the original character. This matches
punctuation to punctuation rather than capital height. Each assigned character
can have its own bounded size multiplier. If either outline cannot be measured,
the font-wide calibration applies.

The reference baseline is the line top plus the text font's ascent, regardless
of a fallback's ascent or calibrated size. Missing-character fallbacks use this
alphabetic baseline. Assigned characters with measurable originals also match
the original character's visible bottom: each replacement shifts by its outline
bottom times its rounded pixel size, minus the original's outline bottom times
the text size. This corrects raised artwork while preserving the original
character's descenders. Both renderers use the same fractional offset, so size
rounding cannot introduce a placement mismatch.

Both renderers use the same measurements, rounded pixel sizes and advances
when fitting and drawing. Outline measurements, profiles and font faces are
cached with bounded memory use.

### Fonts with artwork mapped to letters

Some symbol fonts map artwork to ASCII letters and punctuation rather than
standard Unicode symbols such as `♡`. They can be calibrated, but cannot supply
a Unicode character they do not map. Assign their supported letters explicitly
to show that artwork even when the text font contains those letters. Without
an assignment, letter-mapped artwork only appears when the text font lacks the
letter. Settings shows a note for fonts with this limitation.

## Font cache refresh

`fonts/.fonts_cache.json` records Fox Reader's `fox_reader.__version__`, the
calibration schema, and each file's modification time and size. A missing or
different app version invalidates the cache and rereads all fonts. Changed,
new, or invalid entries are rescanned individually when the version matches;
deleted entries are pruned. Cache writes replace the file atomically.

The cache contains no user fallback order or character assignments, so changing
these preferences does not require sanitising all fonts again. Glyph search
catalogs, recent search pages and font faces are cached for efficient reuse.

## Font licensing

Font files have their own licences, separate from Fox Reader's licence. Only
copy a font into `fonts/` when its licence allows your intended use, and check
redistribution rights before sharing font files.

- **SIL Open Font License (OFL):** allows use, sharing and bundling, including
  commercially, subject to the licence's conditions.
- **Free downloads:** may allow only personal use, independent comics, or a
  limited print run. Read the licence included with the download.
- **Font archives:** licences vary by file; check the font's own page and licence.
- **System fonts:** being installed with an operating system does not necessarily
  grant permission to redistribute the files or embed them in distributed work.

Published pages need fonts licensed for that use. Choose optional fonts by the
lettering style you need and the rights their licences grant.
