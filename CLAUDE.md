# RotatE one-class SVM — KG garbage collection

Flags likely-wrong `biolink:treats` edges in ROBOKOP using a one-class SVM over
RotatE embeddings, and measures honestly whether that works.

## Before doing anything

Read the skill that matches the task — they carry the constraints that are not
visible in the code:

| task | skill |
|---|---|
| run/extend a stage, add a predicate | `.claude/skills/kg-anomaly-pipeline` |
| submit anything to the cluster | `.claude/skills/slurm-run` |
| read, quote or present an AUC | `.claude/skills/read-eval-results` |
| rebuild a figure or the deck | `.claude/skills/build-deck` |

## Hard rules

1. **No compute on a login node.** Everything goes through `sbatch`, and ask
   before submitting — it spends shared allocation.
2. **Generated data is read-only.** `config.refuse_overwrite` guards every
   stage output. Re-run under a new `--out-prefix`/`--tag`; do not delete a
   previous result to clear the way.
3. **No hardcoded paths.** Every external path comes from `config.py`, which
   reads `ROTATE_EMB` / `ROTATE_GRAPH` / `ROTATE_DEG_SRC` / `ROTATE_OUT` with
   Longleaf defaults.
4. **Baselines stay in the evaluation.** `concat = 0.449` means nothing without
   `distance = 0.831` and chance `= 0.500` beside it.
5. **The negative result is the result.** The trained detector loses to the
   untrained RotatE distance. Do not tune to make that go away, and do not
   report around it.

## Layout

```
config.py          all external paths, one place
rotate.py          embedding loading + the rotation algebra, written once
extract_edges.py   stage 1
degrees.py         stage 2
detect.py          stage 3 — the detector, writes gc_candidates_*.tsv
plot_explain.py    stage 3 figure
evaluate.py        stage 4 — treats vs contraindicated_in
replicate_original.py / plot_rbf.py   stage 4b — faithful port of the original
legacy/            superseded, kept as a cross-check; not an entry point
slurm/             env.sh + one .sbatch per stage
results/           the small committed outputs of the completed run
out/               working directory (large files gitignored)
```

`rotate.py` exists so detect and evaluate cannot disagree about the node layout
(`[re | im]`, 256 complex dims in 512 real columns) or the rotation. Add shared
math there, not in a stage.
