---
name: read-eval-results
description: Interpret the stage-4 evaluation outputs of this repo correctly. Use when reading evaluation_summary.tsv or _meta.json, quoting an AUC, comparing the detector against a baseline, writing up or presenting the result, or when someone calls the concat score a bug or proposes tuning it upward.
---

# Reading the evaluation

`out/evaluation_summary.tsv`, one row per model, `kind` marking `svm` vs
`baseline`. The completed 10-seed run:

| model | kind | ROC AUC | PR AUC |
|---|---|---|---|
| distance | baseline | **0.831 ± 0.004** | 0.964 |
| residual | (dropped from the current pipeline) | 0.708 ± 0.006 | 0.940 |
| concat | svm | **0.449 ± 0.007** | 0.832 |
| degree | baseline | 0.434 ± 0.005 | 0.842 |

## What this says

**The one-class SVM on `concat(h, t)` does not work. 0.449 is below chance**,
and statistically indistinguishable from the degree-only control at 0.434 — the
SVM is doing little more than noticing which endpoints are poorly connected.
Meanwhile `||h ∘ r − t||`, RotatE's own score with no training at all, gets
0.831.

That is the finding. It is **not** a bug to be tuned away, and `nu` has already
been selected per seed on a held-out validation split (it lands on 0.5–0.7 for
concat, i.e. the grid's own way of saying the fit is not informative). Report it
as the result: on these embeddings, the trained detector is beaten by the
untrained distance it was supposed to improve on.

## Three traps

1. **PR AUC looks wonderful and means almost nothing here.** The test pool is
   ~1,932 positives against ~12,817 negatives, so the anomaly class prevalence
   is **0.869** — that is the PR AUC floor. `concat`'s 0.832 is *below* it.
   Always quote PR AUC next to `prevalence` from `_meta.json`, or use ROC AUC,
   which is invariant to the ratio.
2. **Never drop the baseline rows.** `concat = 0.449` is only interpretable
   beside `distance = 0.831` and chance = 0.500. Alone it reads as a failed run
   rather than a finding. `--no-baselines` exists but costs interpretability;
   the baselines add no SVM fit and essentially no runtime.
3. **The error bar must come from the positive splits.** It is the spread over
   10 independent 60/20/20 splits. An earlier version held the split fixed and
   resampled negatives, which measured sampling noise (±0.003) rather than
   anything about the method. If a change makes the error bar shrink
   dramatically, suspect the variance source first.

## The seen/unseen breakdown

`out/evaluation_seen_unseen.tsv` splits held-out positives by whether the SVM
saw both endpoints during fitting. Every model drops sharply on `has_unseen`
(concat 0.520 → 0.225; distance 0.881 → 0.685). So what skill exists is
substantially memorisation of familiar nodes, and performance on genuinely new
chemicals or diseases is worse than the headline implies. Quote both.

## Why the ceiling exists

Panel 4 of `explain_treats.png` is the explanation: the distribution of
`||h ∘ r − t||` for `treats` and for `contraindicated_in` largely **overlap**.
RotatE was trained on a graph where both relations connect the same chemicals
to the same diseases, so the embedding has limited means to separate them. No
choice of detector on top of these vectors escapes that.

## The negatives

`biolink:contraindicated_in`, never fitted on, scored through the **treats**
rotation (the question is "does this pair look like a treats edge?"). They are
type-matched hard negatives: same node types, same direction, opposite clinical
meaning. They are not random corruptions, and the numbers are not comparable to
a paper that samples negatives randomly — such a paper's AUC is inflated
relative to this.

## Cross-check

`results/cross_check/evaluation_auc.tsv` is an independent earlier
implementation (`legacy/evaluate_ocsvm.py`, Slurm job 7261544) of the same
comparison with a 1:1 balanced test and a different variance source. It agrees
within 0.01 on all four models. Cite it when the result is challenged.
