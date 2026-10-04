# Fox Reader — Terms of Use

_Last updated: 2026-09-26. These terms may change; continued use after a
change means you accept the new terms._

## 1. Acceptance and scope

By downloading, installing, running, or hosting Fox Reader (source,
self-built, or a released binary) you agree to these terms **and** to the
licenses in `LICENSE` (project code) and `NOTICE` (third-party material).
If you do not agree, do not use the software. You must be able to form a
binding agreement (or have permission from someone who can).

These terms cover the software. They do not replace any upstream license:
model weights, datasets, fonts, libraries, and online APIs each keep their
own terms, listed in `NOTICE` §§1–4 and linked from `/setup` next to each
download.

## 2. License summary (short version — LICENSE and NOTICE control)

- Fox Reader's own code: **AGPL-3.0-or-later**. You may run, study, share,
  and modify it. If you distribute binaries **or** run a modified version as
  a service, you must offer the corresponding source under the same license.
- A separate **paid commercial license** for the project's own code may be
  available from the copyright holder. It never covers third-party weights,
  data, fonts, or libraries.
- Contributions are accepted under AGPL-3.0-or-later with permission to
  relicense (see NOTICE §0). Donations grant no license exceptions.

## 3. Models and datasets — your responsibility

Model weights are **not shipped** with this repository (see `.gitignore`:
`/models/`, `*.pt`, `*.onnx` are excluded). They download from upstream
vendors when **you** run `/setup`. That download is an agreement between
you and the vendor. Before enabling a model, read its terms; the sharp
edges are:

- **Text-seg weights — research code is MIT** (`juvian/Manga-Text-Segmentation`),
  but the weight file itself declares no license; treat the weights as
  academic-only.
- **Bubble segmentation weights** — Apache 2.0 card
  (`iamokish/manga-bubble-segmentation-pytorch`), but trained on
  academic-only Manga109 lineage: do not assume commercial rights.
- **Gemma weights** — Google Gemma Terms of Use (acceptance + prohibited
  uses + carry-the-terms downstream).
- **VNTL weights** — Llama 3 Community License (Acceptable Use Policy,
  attribution, 700M-MAU clause).
- **MS92 data lineage** — credit `"Copyrighted by Minshan Xie"` wherever
  required (see NOTICE §4).

If a vendor term and these terms conflict for that vendor's file, the
vendor's term wins for that file.

## 4. Manga copyright rules

Fox Reader is a tool for processing manga **you own or hold rights to
process** (your scans, licensed works, public-domain works, or works whose
authors permit it). The tool grants you no rights in anyone's manga:

- Do not use it to strip, re-letter, or redistribute manga you have no
  right to copy — including uploading others' pages to shared endpoints,
  public Colab runtimes, or any service that stores them.
- If you publish pages or results derived from **Manga109-s** material,
  obey its conditions (attribute Manga109-s, no redistribution, never sell
  images with results, max 20% of whole pages per volume).
- Credits required by NOTICE §6 (papers, dataset notices) apply to academic
  or public write-ups of your results.

**Endpoints you configure are your responsibility.** `/translate/custom`
exists so you can point Fox Reader at infrastructure **you** run or are
authorised to use: a model server on your own machine, your own VPS, or a
third-party endpoint whose operator permits this use. Configuring a
destination is choosing who receives your pages and text — sending them to
a shared, public, or unauthorised service is your breach of this section,
not a feature of the tool. Fox Reader does not inspect, validate, rank, or
vouch for any endpoint you add, and ships with no endpoint of its own; the
same applies to the DeepL and JPDB keys you supply (§5).

**These rules are stated, not enforced.** Fox Reader does not inspect your
pages, verify ownership, meter how much of a volume you publish, or phone
home, and no part of the pipeline (preview, inpaint, save, folder load)
gates on this section — a local tool cannot check any of it, and a check
that cannot be correct is worse than an honest notice. The `/setup` page
therefore states this section's substance once, on first run, and keeps it
reachable under **Terms & Licenses**. Compliance is yours.

## 5. Online translation services

DeepL, JPDB, and custom endpoints run under **their providers' terms** with
**your** API keys and **your** billing. You are responsible for their
acceptable-use policies, rate limits, and for any content you send them
(including encrypted content — encryption hides text from the network, not
from the endpoint, which must decrypt it to translate).

Every such destination is one you chose and configured; none ships enabled,
and the project operates no translation service. Whether you are entitled to
send a given page or line to a given endpoint is a §4 question, and the
answer is yours to know.

## 6. Prohibited uses

- Breaking the law; infringing copyright, trademark, or privacy rights.
- Using uncensored models to produce illegal content (including CSAM —
  which we report).
- Misrepresenting modified versions as the original project; stripping
  license, credit, or attribution notices.
- Circumventing model-vendor gates (e.g. automating acceptance of gated
  dataset/model terms, or stripping Gemma/Llama attribution and use
  restrictions when redistributing those weights).
- Holding yourself out as the project for phishing or malware distribution.

## 7. Copyright complaints (DMCA-style notice procedure)

This repository hosts **no manga pages and no model weights** — weights
fetch from upstream vendors at setup time — so most complaints will concern
either (a) this project's own files, or (b) something a third party did
with the tool.

**Designated contact for copyright notices**

    Email:      user@foxreader.com
    Subject:    "Copyright notice — Fox Reader" (or "Counter-notice")
    Repository: https://github.com/iamokish/foxreader
                (open an issue for anything that need not stay private)

Email is the preferred channel and the fastest route for both notices and
counter-notices; the repository remains available for public matters. No
postal address is published because the project operates no hosted service
and no user-content platform — if formal service requires one, ask by email
and it will be provided.

A useful notice contains: (1) identification of the copyrighted work,
(2) where it appears (URL, file path, release tag), (3) your contact
details, (4) a good-faith statement that the use is unauthorized, and
(5) a physical or electronic signature. Counter-notices should contain
the same plus consent to jurisdiction. Valid, complete notices are
reviewed promptly; material found infringing is removed; repeat
infringers lose access to project channels. Nothing here waives any
rights or limits any defense.

Complaints about a *third party's* use of the tool — a scanlation site, a
redistributed release, a public endpoint — have to go to that party or to
whoever hosts them: the project has no access to anyone's installation, no
telemetry, and no ability to disable a copy already downloaded (§4).

## 8. No warranty, limited liability

The software is provided **"as is"**, without warranty of any kind
(see LICENSE §§15–17, which control). Machine-learning outputs
(translations, segmentation, inpainting) may be wrong; verify anything
that matters. To the maximum extent permitted by law, the contributors
are not liable for damages arising from use or inability to use the
software, including mistranslations or data loss.

## 9. Changes and contact

Terms and model terms change upstream without notice; re-check before
each release you ship. Questions about licensing, commercial terms, or
these Terms: example@fox-reader.com, or through the repository above.
Copyright notices go to the §7 contact.
