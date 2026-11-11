# Garbage-collecting a knowledge graph with a one-class SVM over RotatE embeddings

ROBOKOP contains `biolink:treats` edges that should not be there. This repo
tries to find them without labels: fit a one-class SVM on the RotatE embeddings
of every `treats` edge, flag the ones that fall outside the learned region, and
then — the part that is usually skipped — check against held-out known-bad edges
whether the flags mean anything.

**They largely do not.** The headline result is negative, and it is the point.

## Result

10 independent 60/20/20 splits of 9,657 `biolink:treats` edges. Negatives are
25,633 `biolink:contraindicated_in` edges, never fitted on, scored through the
*treats* rotation. Error bars are the spread across splits.
(`results/evaluation_summary.tsv`)

| model | trained? | ROC AUC | PR AUC |
|---|---|---|---|
| `distance` — `‖h ∘ r − t‖`, RotatE's own score | no | **0.831 ± 0.004** | 0.964 |
| `residual` — OCSVM on `h ∘ r − t` *(dropped; see below)* | yes | 0.708 ± 0.006 | 0.940 |
| `concat` — OCSVM on `concat(h, t)` | yes | **0.449 ± 0.007** | 0.832 |
| `degree` — endpoint degrees only (control) | yes | 0.434 ± 0.005 | 0.842 |

Chance is 0.500. So:

- The one-class SVM on `concat(h, t)` — the original method — is **below
  chance**, and indistinguishable from a control that sees nothing but how
  well-connected the two endpoints are. It is detecting node rarity, not
  wrongness.
- An **untrained** baseline, RotatE's own triple score, reaches 0.831 and beats
  every trained detector here.
- The PR AUCs are a trap. Anomaly-class prevalence in the test pool is **0.869**,
  which is the PR AUC floor; `concat`'s 0.832 is *below* it. Quote ROC AUC, or
  quote PR AUC next to prevalence.

Splitting held-out positives by whether the SVM had seen both endpoints during
fitting (`results/evaluation_seen_unseen.tsv`) shows most of what skill exists is
memorisation: `concat` 0.528 → 0.219 and `distance` 0.886 → 0.671 when an
endpoint is new. `concat` on genuinely new endpoints is far *below* chance.

### Why the ceiling is there

The distributions of `‖h ∘ r − t‖` for `treats` and for `contraindicated_in`
overlap substantially. RotatE was trained on a graph where both relations join
the same chemicals to the same diseases, so the embedding has little means to
tell them apart — and no detector built on top of these vectors escapes that.
Panel 4 of the stage-3 figure shows it directly.

### What does work: the type rule

The most anomalous edges are not subtle. They point at **procedures and imaging
agents**, not diseases — Besilesomab → *Radionuclide Imaging*, indocyanine green
→ *Lymphatic imaging*, air polymer-type A → *Fertiloscopy*, crilanomer → *Wound
dressing* (`results/replication_rbf/top50_flagged_rbf.tsv`).

A `treats` edge must point at a `biolink:Disease` or a
`biolink:PhenotypicFeature`. That is a **type rule, true independent of any
model**, and `detect.py` evaluates it as the `type_rule_violation` column.
Edges that are both flagged *and* violate the type rule are defensible
deletions without trusting the SVM at all, and that intersection — not the SVM
score — is the usable garbage-collection output.
`results/replication_rbf/check_flagged.cypher` confirms them against upstream
ROBOKOP rather than the filtered subgraph.

Measured on the committed stage-3 run (`results/summary_treats.tsv`), at
`nu=0.05`:

| | count | share of 9,657 |
|---|---|---|
| flagged by the SVM | 482 | 5.0% |
| violating the type rule | 350 | 3.6% |
| **both — deletable without the model** | **98** | **1.0%** |

The type-violation rate is **20.3% among flagged edges against 2.8% among
unflagged**, a 7.4x enrichment, so the detector is not noise. But it found only
98 of the 350 violations and missed 252, and the type rule alone finds all 350
at no cost — which is the case for running the rule and treating the SVM as a
secondary filter, not the reverse.

Two caveats on the 482. `score` correlates **0.875** (Spearman) with
`min_degree`, and median degree is 3 among flagged edges against 77 among
unflagged, so much of the flagging is node rarity rather than wrongness; sort by
`type_rule_violation` before `score`. And the rule as implemented checks the
**object side only**. Adding the subject rule (the treating entity must be a
chemical: 61 further violations, mostly `BiologicalProcess` and `Pathway`) and a
self-loop check (1, a compound recorded as treating itself) raises the
model-independent total to **411 distinct edges**, 4.3% of the predicate. That
extension is not yet in `config.py`.

## Outputs

Every deliverable is a TSV or CSV. The figures that the plotting scripts produce
are diagnostics, not products.

| file | what it holds |
|---|---|
| `out/gc_candidates_<slug>.tsv` | **the deliverable.** Flagged edges, most anomalous first: endpoints, both node types, both degrees, `min_degree`, `type_rule_violation`, `primary_knowledge_source`, `score`, `rotate_dist_l2`, `rotate_dist_l1` |
| `out/ranked_<slug>.tsv` | the same columns for *every* edge, not just the flagged ones — use it to pick a different threshold |
| `out/top50_<slug>.tsv` | the 50 most anomalous, for hand-reading |
| `out/summary_<slug>.tsv` | one row: flag count and rate, score-vs-degree Spearman, median degree flagged vs unflagged, type-rule violation rate on each side |
| `out/evaluation_summary.tsv` | one row per model: ROC/PR AUC mean, std, min, max, and `kind` (`svm` or `baseline`) |
| `out/evaluation_per_seed.tsv` | the same, per seed, with the `nu` chosen on that seed's validation split |
| `out/evaluation_seen_unseen.tsv` | per model and seed, AUC split by whether both endpoints were in the fit set |
| `out/evaluation_meta.json` | the run's configuration, including `prevalence` — needed to read any PR AUC |
| `out/original_<slug><tag>/OneClassSVM_Results_<slug>.csv` | the replication's per-edge ±1 predictions |
| `out/original_<slug><tag>/grid_search.csv` | the 32-point grid, scored |
| `out/census.txt` | predicate census over the source graph, from stage 1 |

To triage candidates, sort `gc_candidates_<slug>.tsv` by `type_rule_violation`
first and `score` second: those rows are wrong by the type rule regardless of
what the model thinks.

## Inputs

None of these live in the repo — they are hundreds of megabytes to two
gigabytes each. Point `config.py` at them with `ROTATE_EMB`, `ROTATE_GRAPH` and
`ROTATE_DEG_SRC`; the defaults are the Longleaf paths the committed run used.

**`ROTATE_EMB` — the RotatE embeddings** (1.9 GB). A RotatE model over the
ROBOKOP subgraph: 931,502 entities, 116 relations, `embedding_dim` 512, so 256
complex dimensions, trained with `gamma=12.0`. The export is epoch 50 of an
Azure ML run, at `val_auc` 0.834 / `val_mrr` 0.053. `node_embeddings.npy` is
laid out `[re | im]`; `relation_rotations.npy` is `[cos(phase) | sin(phase)]`,
which is what `rotate.py` consumes. The directory carries a `manifest.json` with
the source checkpoint, its SHA-256, and a SHA-256 per exported file — check
those before trusting a copy.

**`ROTATE_GRAPH` — `robokop_30fd_filtered`** (7.6 GB). `subgraph_edges.jsonl`,
read once by stage 1, plus `training_node_types.tsv`, which supplies the node
types the type rule is evaluated against.

**`ROTATE_DEG_SRC` — a flat TSV of the whole graph** (1.1 GB,
`subject/predicate/object`). Generated from the same subgraph as above, minus
1,262 documented dangling self-loops. Stage 2 reads it only to count global node
degree, which is needed twice downstream: as context on every flagged edge, so a
flag can be read as "rare node" rather than "wrong edge", and as the degree-only
control in stage 4. A JSONL pass would give the same counts; the TSV is used
because it is a tenth the size.

## Pipeline

```bash
sbatch slurm/extract.sbatch                        # 1. edges.tsv, census.txt
sbatch slurm/degrees.sbatch                        # 2. degrees.tsv
sbatch slurm/detect.sbatch biolink:treats 0.05     # 3. gc_candidates + figure
sbatch slurm/evaluate.sbatch --seeds 10            # 4. the table above
sbatch slurm/replicate.sbatch --kernels rbf --tag _rbf   # 4b. original method
sbatch slurm/plot_rbf.sbatch original_treats_rbf
```

Stages 1–2 run once; their outputs are read-only inputs to the rest. Every
stage refuses to overwrite an existing result, and validates its inputs in the
first second rather than after an hour of compute.

Submit from the repo root. Nothing is run on a login node. See
`.agents/skills/anomaly/slurm-run`.

### Development

```bash
make setup     # uv sync --dev
make test      # ruff check + ruff format --check, then pytest -q
make skills    # refresh .claude/skills symlinks from .agents/skills
```

Conventions follow [KGB-CIA](https://github.com/UNC-RENCI-KGAI/KGB-CIA): `AGENTS.md` rather than
`CLAUDE.md`, canonical skills under `.agents/skills/<category>/<skill-name>/`, ruff at line
length 100 with rules `E F I W UP B`. This repository is standalone and is not merged into
KGB-CIA. See [`AGENTS.md`](AGENTS.md).

The tests need no embeddings, no graph and no network.

### Configuration

No paths are hardcoded. `config.py` reads, with Longleaf defaults:

| variable | what |
|---|---|
| `ROTATE_EMB` | RotatE artifacts: `node_embeddings.npy`, `node_index.tsv`, `relation_index.tsv`, `relation_rotations.npy` |
| `ROTATE_GRAPH` | `robokop_30fd_filtered/`: `subgraph_edges.jsonl`, `training_node_types.tsv` |
| `ROTATE_DEG_SRC` | graph-wide `edges.tsv` for the degree count |
| `ROTATE_OUT` | output directory (default `./out`) |

Inputs are large and not in the repo: embeddings are 1.9 GB (memory-mapped, so
only indexed rows are read), the source graph larger still.

## Layout

```
config.py rotate.py          paths; embedding loading and rotation algebra
extract_edges.py degrees.py  stages 1-2
detect.py plot_explain.py    stage 3 — the detector and its diagnostic figure
evaluate.py                  stage 4 — treats vs contraindicated_in
replicate_original.py        stage 4b — faithful port of the pre-RotatE method,
plot_rbf.py                  with GraphSAGE swapped for RotatE
legacy/                      superseded, kept as a cross-check
slurm/                       env.sh (all site config) + one .sbatch per stage
tests/                       rotation algebra and the two guards, no data needed
results/                     small committed outputs of the completed run
.agents/skills/anomaly/      how to run and read this correctly
.claude/skills/              generated symlinks (make skills)
```

## Scope

Only `concat(h, t)` is fitted. The `residual` feature set was dropped from the
pipeline to keep it to the one method under study; its measured result is kept
in the table above and in `results/` because removing a model from the code is
not a reason to hide what it scored.

`distance` and `degree` remain in the evaluation as **reference rows that are
never fitted**. They cost no training time and `concat = 0.449` is not
interpretable without them.

`legacy/evaluate_ocsvm.py` is the first implementation of this evaluation and
the origin of its design. `evaluate.py` kept the design and corrected the
protocol: exact scoring instead of 1:1 sampled test draws, and the error bar
moved from negative resampling (which measured ±0.003 of sampling noise) to the
positive split. The two agree within 0.01 on all four models —
`results/cross_check/` — which is why the old one is still here.

## Reproducing

`results/` holds the stage-3 detector run, the stage-4 evaluation, the
replication's grid search, flagged list and Cypher checks.

Stage 3 was regenerated on 2026-11-06 under the current code (Slurm 8401579 /
8401581 / 8401583), so `gc_candidates_treats.tsv`, `top50_treats.tsv` and
`summary_treats.tsv` now carry the `type_rule_violation` column and are sorted by
the score the pipeline actually computes. The earlier outputs predated the
removal of `residual` and were sorted by a score no longer produced; they are
gone. Stages 1–2 reproduced their inputs exactly — `edges.tsv` is byte-identical
to the earlier run and `degrees.tsv` matches on content, differing only in row
order, which is immaterial because it is consumed as a lookup.

`ranked_treats.tsv` (all 9,657 edges, 1.5 MB) stays uncommitted by `.gitignore`;
one `sbatch slurm/detect.sbatch` regenerates it alongside the rest.

The stage-4 numbers are unaffected — per-model scores are computed
independently, so dropping a model does not change the others.

Every random draw is explicitly seeded — `default_rng(1000 + seed)` for the
splits, `default_rng(0)` for the node sample in the figure — and `OneClassSVM`
with an rbf kernel is deterministic, so the same inputs under the pinned
dependencies give the same numbers. Two things are expected to vary and do not
matter: `degrees.tsv` row order, because it is written from a Python set whose
iteration order depends on hash seeding and is consumed as a lookup; and the
exact bytes of the PNG, which depend on the matplotlib build.
