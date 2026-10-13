# Phase 7 — Evaluation, Benchmarking & Documentation (Weeks 13–15)

**Owner:** whole team. **Goal:** honest numbers, a comparison against the six papers you reviewed, and a final report/demo that doesn't overclaim.

## 7.1 Metrics

Compute these on **both** validation and unseen-test splits, per field and overall:
```python
import evaluate
cer = evaluate.load("cer")
wer = evaluate.load("wer")

def exact_match_accuracy(preds: list[str], refs: list[str]) -> float:
    return sum(p == r for p, r in zip(preds, refs)) / len(refs)
```

## 7.2 Benchmark table — build this to directly extend your literature review's comparison table

| System | Dataset scale | CER | WER | EMA | Multi-field? | Multilingual? |
|---|---|---|---|---|---|---|
| Biswas et al. (2026) | 400,000+ | — | — | 90.20% overall | Yes (5 fields) | No |
| Ullah et al. (2025) | 580 | 8.7% | 12.5% | 81.3% | No | No |
| Ali et al. (2024) | ~1,000→9,920 | 1.4% (val) / 13.5% (unseen) | — | — | No | No |
| **Your system** | *(fill in)* | *(val / unseen)* | *(val / unseen)* | *(val / unseen)* | Yes (5 fields) | Yes (2 languages) |

Filling in your own row honestly — including reporting a worse number than the literature where that's actually true — is more credible to a grader than an unsupported claim of beating everything. Your genuine differentiator is breadth (structured multi-field + multilingual + human-review flagging), not necessarily raw CER beating a system trained on 400,000 real prescriptions.

## 7.3 Ablation — show the post-processing layer's actual value

Report CER **with and without** the Phase 4 domain-aware correction layer, isolated — this directly demonstrates (with your own numbers) the same effect your literature review noted qualitatively across multiple papers, and is an easy, high-value experiment to include.

## 7.4 Final report structure

Mirror your synopsis's section structure (it's already well organized) but add:
- An explicit "Deviations from the original synopsis" section: state the YOLO-vs-Mask-R-CNN swap, one-shared-model-vs-five, two-language MVP, and the revised data strategy, each with your reasoning. Graders generally respond well to justified, documented trade-offs — much better than either silently under-delivering or awkwardly claiming everything was achieved as originally scoped.
- The benchmark table above, with your ablation results.
- Limitations, stated plainly (small dataset relative to Biswas et al., two languages not full multilingual coverage, demo-scale deployment not production-hardened security).

## 7.5 Demo prep

- Pre-select 3–4 real example prescriptions that show the system working well, plus one that shows a low-confidence field correctly getting flagged for human review — a live "and here's what happens when it's uncertain" moment is more convincing than only showing perfect runs.
- Have a recorded video backup in case live GPU/network demo hiccups happen during your actual presentation slot.

## Definition of done for Phase 7

- [ ] CER/WER/EMA computed per field, per language, on both validation and unseen-test splits
- [ ] Benchmark table filled in against the six reviewed papers
- [ ] Ablation (with/without post-processing) run and reported
- [ ] Final report written, including an explicit deviations-from-synopsis section
- [ ] Demo rehearsed, video backup recorded
