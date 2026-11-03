---
name: build-deck
description: Rebuild the figures and the PowerPoint deck for this project. Use when asked to regenerate a figure, update slides, refresh the presentation, or report results to an audience.
---

# Figures and deck

Order matters: a figure is built from a stage's saved arrays, and the deck is
built from the stage's TSVs.

`node` is not on PATH by default on Longleaf — load it first:

```bash
module load nodejs/24.16.0
```

```bash
# figure for the main detector (stage 3 must have run)
sbatch slurm/detect.sbatch biolink:treats 0.05     # writes viz_*.npz, then the png
# figure for the faithful replication
sbatch slurm/plot_rbf.sbatch original_treats_rbf
# deck
cd presentation && npm install && node build_deck.js [out_prefix]
```

## The rule the deck already follows — keep it

`build_deck.js` **reads every number out of `out/*_summary.tsv` at build time**.
None are typed into the slides. That is deliberate: it makes it impossible for
the deck to drift from the run that produced it. If you need a number the TSVs
do not contain, add it to a stage's output and read it — do not type it into the
deck.

With no argument the builder picks the most recently modified `*_summary.tsv`.
Pass the prefix explicitly when more than one evaluation exists, or the deck
will silently describe whichever ran last.

## Figures

- `plot_explain.py` → `out/explain_<slug>.png`, four panels, from
  `out/viz_<slug>.npz`. Panel 4 (the `treats` vs `contraindicated_in` distance
  overlap) is the one that explains the ceiling; keep it in any shortened
  version.
- `plot_rbf.py` → `out/explain_rbf_<run>.{png,pdf,svg}` for the replication run.
  It **refits** the selected model (rbf, nu=0.2, gamma=scale) because the run
  wrote only ±1 labels and the figure needs the decision function. If the
  replication's selected parameters change, change them at the top of
  `plot_rbf.py` too — they are not read from the run.

In both, PCA is a viewing device only. The SVM is fit in the full 1,024-dim
space; say so in any caption, since a 2-D plot invites the reader to think the
projection produced the flags.

## Palette

Both figures use a validated 3-slot categorical palette on a light surface:
`#2a78d6` blue, `#eb6834` orange, `#1baf7a` aqua, with `#b9b8b2` as the
recessive gray for the unflagged mass. Flagged is always orange, kept is always
gray. Keep it consistent across figures — the deck puts them side by side.

## Theme dependency

`build_deck.js` requires `vendor/apply_theme.js`, vendored into this repo. It
previously loaded that file from a machine-local path under
`~/.claude/skills/synced/<uuid>/`, which broke for anyone else who cloned.
Do not reintroduce an absolute path outside the repo.

## When presenting

The honest headline is that the trained detector lost to the untrained baseline.
Read `read-eval-results` before writing a claim about performance — especially
before quoting a PR AUC, which is near its floor here.
