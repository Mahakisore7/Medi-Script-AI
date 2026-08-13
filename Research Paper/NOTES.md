# Notes on this draft — read before review

This folder contains a full IEEEtran LaTeX build of the Medi-Script-AI paper,
compiled cleanly from the provided project write-up plus the "Complete
Project Status Report." This file documents every substantive editorial
decision so it can be checked quickly rather than re-derived.

## Contents

```
main.tex                       paper source (IEEEtran, conference style)
references.bib                 all 7 sources, verified against primary pages
main.pdf                       pre-compiled output (6 pages), included for convenience
figures/pipeline_architecture.pdf   Figure 1, generated from make_pipeline_fig.py
figures/results_metrics.pdf         Figure 2, generated from make_results_fig.py
figures/make_pipeline_fig.py        regenerate Figure 1 (matplotlib)
figures/make_results_fig.py         regenerate Figure 2 (matplotlib)
```

## How to compile

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires `texlive-publishers` for `IEEEtran.cls`/`IEEEtran.bst` (not in a
base TeX Live install) — everything else is in `texlive-latex-recommended`
and `texlive-latex-extra`. Regenerating the figures needs `python3-matplotlib`.

## Before this goes further, please fill in

- **Author name(s), affiliation, email** — placeholder text at the top of
  `main.tex` (search `TODO`). Nothing else depends on this.
- **CPU-only inference benchmark for YOLOv8-nano.** §IV-B originally claimed
  "under 3ms on CPU." The only hard number in the status report (2.5ms/image)
  was measured on a Colab **T4 GPU**, not CPU — and typical published
  YOLOv8-nano CPU numbers at 640×640 are tens of ms, not single digits. Since
  the paper's edge-deployment argument leans on this figure, I softened the
  claim to "designed for constrained hardware" and flagged the real CPU
  number as an open measurement (§IV-B, §V-D, §VI-A) rather than assert a
  number I couldn't verify. Worth an actual Raspberry-Pi benchmark before
  submission.
- **Last-page column balance.** IEEEtran's own compile note applies: the
  two columns on page 6 aren't hand-balanced. Cosmetic, standard camera-ready
  step, didn't seem worth doing before content is final.

## Corrections applied to the source material

**1. IOM citation (per your note).** Year corrected to 2007; editors set to
Aspden, Wolcott, Bootman, and Cronenwett (Eds.), matching the report's actual
front matter. I also tightened the abstract/intro claim to what the report
actually says — the \$3.5B figure is *hospital drug-related-injury costs*
specifically, not medication errors generally.

**2. Reference list now matches your 7 sources exactly.** The original
draft's reference list cited Redmon & Farhadi's YOLOv3 paper (unused
anywhere in the text — the system is built on YOLOv8) instead of an
Ultralytics YOLOv8 citation. Replaced with the software citation you
specified (Jocher, Chaurasia, Qiu, 2023), pulled from the repo's own
`CITATION.cff`.

**3. Reference #7 was mischaracterized in the original draft.** I fetched
`github.com/JonSnow1807/Medical-Prescription-OCR` directly to write its
BibTeX entry, and found it's a **Donut-based** project (it's the actual
source of the `chinmays18/medical-prescription-dataset` model discussed in
§III-C), not the YOLOv5-only project the original draft's §3.4 described it
as. I moved the citation to §III-C where it factually belongs, and rewrote
§III-D to make the same architectural point (naive detect-then-OCR pipelines
inherit generic OCR's handwriting failures) without the incorrect
attribution. If you had a *different* YOLO-only repo in mind for that
point, let me know and I'll re-cite it properly.

**4. All AAAI / ACM MM / ECCV proceedings details (volume, pages, DOI)** were
pulled from the publisher pages, not reconstructed from memory — TrOCR
(AAAI-23, 37(11):13094–13102), LayoutLMv3 (MM '22, pp. 4083–4091), Donut
(ECCV 2022, LNCS 13688, pp. 498–517).

## The one substantive rewrite: Abstract, Results, and Conclusion

The project status report changes what the paper can honestly claim, so I
rewrote these three sections rather than just reformatting the original
draft. Specifically:

- The original abstract claimed the hybrid pipeline "significantly
  outperform[s]" other approaches on real prescriptions. The status report
  shows the opposite on the two real-image runs you documented: the
  end-to-end pipeline returns either an empty record (safe, but not a
  success) or a fluent hallucination (a real commercial product name, not
  the actual handwriting). Only the YOLO detection stage has a real,
  positive, load-bearing number (86.1% mAP50).
- Rather than drop the negative result, I built it into the paper as a
  proper contribution: §VI-C now reports two case studies (in-distribution
  and out-of-distribution) diagnosing *exactly* how and why Stage 2 fails,
  which I think is more convincing than an aggregate accuracy number would
  be at this stage. I generalized the patient-name example rather than
  reproducing it — no real name appears in the paper.
- I also fixed one thing that would otherwise cause quiet confusion later:
  the "Vivek" extraction in your report worked because of a mode-selection
  bug that routed the image through a PaddleOCR fallback, not through
  YOLO+TrOCR. I recorded this as a one-paragraph methodological note in
  §VI-C (per-stage provenance logging) rather than as a pipeline result,
  since attributing it to the proposed architecture would've been incorrect.
- The new mAP50-95 (46.2%), training time, model size, and inference-speed
  figures from the status report are now in Table II / Figure 2.
- Conclusion and Future Work now follow directly from this: the three-phase
  plan (fine-tune TrOCR on 200–300 real crops → expand YOLO to 6 classes →
  re-run the case studies to confirm the hallucination is actually fixed)
  uses the specifics from your report — the training script already
  existing, the ~30 min/run cost on a T4, the ~100-image annotation target.

If you'd rather keep the original, more optimistic framing for a specific
venue's expectations, the fix is local to the Abstract, §VI, and §VII —
everything else in the paper (Introduction through §V, the architecture
description) doesn't depend on this framing choice.

## Update: structure + new architecture figure

- **Problem Statement is no longer a standalone section.** The record schema
  ($R$) and its formal setup now close out the Introduction (§I); the
  five-item constraint list was folded into two sentences there (it mostly
  duplicated §IV Challenges, so I didn't repeat it twice). Everything else
  renumbered automatically via LaTeX's `\ref` — no manual number-chasing
  needed if you edit further.
- **New Figure 1** (`figures/model_block_diagram.pdf`, `figures/make_block_diagram.py`)
  replaces the old three-box pipeline sketch. It's a full-width `figure*`
  showing every sub-component named in §III: YOLOv8-nano's
  backbone/neck/head, TrOCR's ViT encoder and constrained-decoding
  autoregressive decoder, the parallel EasyOCR cross-check and RapidFuzz
  agreement node, and Stage 3's lexicon → correction → LM-rescoring →
  confidence-gate chain, with measured operating stats as chips under each
  stage. The old `pipeline_architecture.pdf` / `make_pipeline_fig.py` were
  removed as redundant with this.

## Still waiting on: og_presc.jpg / yolo_presc.jpg

You mentioned a real YOLO-detection example (original prescription +
YOLO-boxed crop) but the two files aren't in the upload folder yet — I
checked. Nothing in this draft depends on them yet. Once they're attached
I'll add them as a proper figure (likely alongside the §V-C case studies,
since that's where the paper already walks through real detection
examples) rather than fabricate placeholder prescription images, since
that would misrepresent what's real experimental output in a paper whose
whole point is being honest about what the system actually does.
