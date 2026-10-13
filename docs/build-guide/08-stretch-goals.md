# Phase 8 — Stretch Goals (only if time remains after Phase 7)

Everything here was explicitly scoped out of the MVP to fit a semester. Don't start any of these until Phases 0–7 are genuinely done — a half-finished stretch goal looks worse in a final report than a clean note that it's future work (and your synopsis already lists most of these under "Future Scope," so treating them this way is consistent with what you already proposed).

- **Additional languages** beyond the two in the MVP — each new language needs its own lexicon and its own evaluated accuracy, not just a mapping table.
- **Per-field model split** — if the shared TrOCR model (Phase 3) is measurably weak on one field type (age/dosage are the likely candidates), train a dedicated smaller model for just that field.
- **Online/stylus capture mode** — following Tabassum et al.'s smartpen concept; requires stylus hardware and is a genuinely separate data-collection effort, not a tweak to the existing pipeline.
- **Doctor-specific personalization** — fine-tuning per clinician's handwriting over repeated use; needs a live-deployment feedback loop that doesn't really exist until after this project ships.
- **Federated/privacy-preserving training** across institutions — a real research topic on its own, not a semester add-on.
- **Full regulatory compliance audit** (HIPAA/GDPR-grade, not just "encryption + access control" best-effort) — needs legal/compliance expertise beyond the team, worth naming as a condition for any real clinical deployment rather than something to attempt in code.
- **Market-basket / co-prescription plausibility checks** (Pavithiran et al.'s idea) — a nice accuracy boost on top of the lexicon correction, but needs prescribing-pattern data you likely won't have collected at MVP scale.

If you do pick one of these up, treat it exactly like a new phase: its own branch prefix, its own definition of done, its own entry in the final report's scope section.
