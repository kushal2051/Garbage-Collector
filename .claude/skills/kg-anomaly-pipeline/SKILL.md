---
name: kg-anomaly-pipeline
description: Run or extend the RotatE one-class-SVM garbage-collection pipeline in this repo. Use when asked to detect anomalous KG edges, flag bad treats edges, produce garbage-collection candidates, re-run a stage (extract, degrees, detect, evaluate, replicate), add a predicate, or when a stage errors with a missing input or a refuse-to-overwrite message.
---

# The pipeline

Five stages. Each reads only what an earlier stage wrote, so a stage can be
re-run alone. All paths come from `config.py`; never hardcode one.

```
1 extract_edges.py  subgraph_edges.jsonl ──> out/edges.tsv, out/census.txt
2 degrees.py        graph edges.tsv      ──> out/degrees.tsv
3 detect.py         1 + 2 + embeddings   ──> out/gc_candidates_<slug>.tsv
                                             out/ranked_<slug>.tsv
                                             out/top50_<slug>.tsv
                                             out/summary_<slug>.tsv
                                             out/viz_<slug>.npz
  plot_explain.py   out/viz_<slug>.npz   ──> out/explain_<slug>.png
4 evaluate.py       1 + 2 + embeddings   ──> out/evaluation_{summary,per_seed,
                                             seen_unseen}.tsv, _meta.json
4b replicate_original.py (+ plot_rbf.py) ──> out/original_<slug><tag>/
```

Stages 1 and 2 are **run once**. Their outputs are inputs to everything else
and are treated as read-only; `config.refuse_overwrite` enforces this. If you
need them rebuilt, move the old file aside deliberately — do not delete it as a
convenience.

## What the method is

Fit a `OneClassSVM(kernel="rbf", gamma="scale")` on `concat(h, t)` for every
edge of one predicate, flag the edges whose `decision_function` is negative.
One class only: there is no label at fit time. A flag therefore means **"this
edge sits in a sparse region relative to the other edges of this predicate"** —
nothing stronger. Turning that into "this edge is wrong" is stage 4's job.

`concat(h, t)` is the only feature set. `distance` and `degree` appear in stage
4 as **reference rows that are never fitted**, and they are what make the concat
number mean anything. Do not drop them from an evaluation; see
`read-eval-results`.

## Garbage collection

`out/gc_candidates_<slug>.tsv` is the deliverable: flagged edges, most anomalous
first, each with the context needed to triage it.

The column that carries the most weight is `type_rule_violation`. A
`biolink:treats` edge must point at a `biolink:Disease` or a
`biolink:PhenotypicFeature` (`config.VALID_TREATS_OBJECT_TYPES`); anything else
is malformed **by definition, independent of any model**. Where the detector and
the type rule agree, the edge is a defensible deletion without having to trust
the SVM. detect.py prints that intersection count at the end.

Always report a candidate alongside `min_degree`. The detector's scores
correlate with node degree, so a flag on a degree-1 node may mean only "rare",
not "wrong" — this is the failure mode that invalidated an earlier node2vec run
on the mf1 graph, and it is why stage 2 exists.

## Adding a predicate

1. Add it to `WANTED` in `extract_edges.py` and re-run stage 1 (it is one pass
   over the jsonl; the `HINTS` substring prefilter must also match it, or the
   lines are skipped before the predicate is ever read).
2. Confirm it is in `relation_index.tsv` — `emb.relation_id()` exits if not.
3. `sbatch slurm/detect.sbatch biolink:<predicate> <nu>`.

If the new predicate is not chemical→disease, `VALID_TREATS_OBJECT_TYPES` no
longer applies and `type_rule_violation` becomes meaningless for it. Make the
rule predicate-aware rather than reading the column as if it still held.

## Running it

Never on a login node — see `slurm-run`. Everything goes through `sbatch`.
