# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Copilot, and others) working in this repository.

Don't add a `CLAUDE.md` anywhere in the repo, even one that only imports this file: Claude Code
reads `AGENTS.md` files directly only as long as no `CLAUDE.md` exists in the repo. (`/init`
creates one, so delete it afterwards and move anything useful into an `AGENTS.md`.)

Conventions here follow [KGB-CIA](https://github.com/UNC-RENCI-KGAI/KGB-CIA) so that the two
repositories read the same way. This one is standalone and is **not** merged into KGB-CIA; it is
not a pipeline stage and has no `shared/schema.py` contract to keep.

## What this is

Garbage collection for a knowledge graph. A one-class SVM is fitted on the RotatE embeddings of
every `biolink:treats` edge in ROBOKOP and flags the edges that fall outside the learned region;
the flagged edges, ranked and annotated, are the output. Stage 4 then measures whether the flags
mean anything, against held-out `biolink:contraindicated_in` edges as type-matched hard
negatives.

The measured answer is that the detector does not work: it scores **0.449** ROC AUC, below
chance and indistinguishable from a degree-only control, while RotatE's own untrained distance
reaches **0.831**. That negative result is the finding, not a bug to be tuned away. Read
[`.agents/skills/anomaly/read-eval-results/SKILL.md`](.agents/skills/anomaly/read-eval-results/SKILL.md)
before quoting any number from this repo.

## Commands

```bash
make setup     # uv sync --dev (falls back to a pip venv if uv is missing)
make lint      # ruff check + ruff format --check
make format    # ruff --fix + ruff format
make test      # lint, then pytest -q
make skills    # refresh the .claude/skills symlinks from .agents/skills

sbatch slurm/extract.sbatch                       # stage 1
sbatch slurm/detect.sbatch biolink:treats 0.05    # stage 3, the detector
sbatch slurm/evaluate.sbatch --seeds 10           # stage 4
```

Ruff line length is 100; rules `E F I W UP B`. Add dependencies with `uv add <pkg>` so
`pyproject.toml` stays in sync. `legacy/` and `out/` are excluded from lint.

The tests run without the embeddings, the graph or a network: they exercise the rotation algebra
against a hand-built stand-in. Keep them that way, since the real artifacts are 1.9 GB and live
outside the repo.

## Architecture

- **Stages are top-level scripts, run in order via `slurm/`:** extract → degrees → detect →
  evaluate, plus `replicate_original.py` as stage 4b. Each reads only what an earlier stage
  wrote, so any stage can be re-run alone.
- **Stage 5 is link prediction, a separate line from stages 1-4:** `kg_prepare` → `pharos_split`
  → `gnn`. It answers a different question — how well a graph predicts its own held-out edges,
  against the Pharos baseline — and shares only `config.py` with the detector. It reads the
  KGB-CIA merged graph, not the ROBOKOP subgraph, via `ROTATE_KG` and `ROTATE_PHAROS`.
- **`kgraph.py` holds the merged graph's layout and the split protocol**, and `sage.py` the
  model and the neighbour sampler, for the reason `rotate.py` exists: written once, so three
  stages cannot drift on what a node id means. Both are numpy/torch only and are tested against
  hand-built stand-ins.
- **Graphs may share one node table or each keep its own.** The default maps both through the
  merged graph's `nodes.jsonl`, so an identical split exists by construction. That costs a
  standalone baseline most of its graph -- for Pharos v319 normalized, 91% of its nodes and
  64.7% of its edges, nearly all of them compounds -- so `kg_prepare.py --nodes-from-edges`
  builds a graph its own table with categories from the CURIE prefix, and `pharos_split.py`
  joins the two on CURIEs. The split file has always held CURIEs, so nothing downstream changes.
- **Unlike KGB-CIA there is no `cli.py` / `main(argv) -> int` contract.** There is no pipeline
  runner to satisfy — stages are submitted individually through `sbatch` — so adopting that
  contract would be ceremony without a caller. This is a deliberate deviation.
- **`config.py` holds every external path**, read from `ROTATE_EMB`, `ROTATE_GRAPH`,
  `ROTATE_DEG_SRC` and `ROTATE_OUT` with Longleaf defaults. No stage hardcodes a path.
- **`rotate.py` holds the embedding loading and the rotation algebra**, written once, so detect
  and evaluate cannot drift on the node layout (`[re | im]`, 256 complex dims in 512 real
  columns). Shared maths goes there, not into a stage.
- **`legacy/` is a frozen record**, not an entry point: the first implementation of the stage-4
  evaluation, kept because it agrees with the current one within 0.01 on all four models. It is
  excluded from lint and format deliberately; reformatting it would defeat the point.

## Design rules

- Agents decide, code transforms: a flag is a suggestion for a human to triage, never an
  automatic deletion. The output is a candidate list.
- The type rule outranks the model. A `treats` edge must point at a `biolink:Disease` or a
  `biolink:PhenotypicFeature`; `detect.py` evaluates that as `type_rule_violation`. Edges both
  flagged and violating are defensible deletions without trusting the SVM at all.
- Report degree alongside every flag. The scores correlate with node degree, so a flag on a
  degree-1 node may mean "rare" rather than "wrong" — the failure mode that invalidated an
  earlier node2vec run on the mf1 graph.
- Baselines stay in the evaluation. `concat = 0.449` means nothing without `distance = 0.831`
  and chance `= 0.500` beside it.
- Every deliverable is a TSV or CSV. The figures are diagnostics; whatever a figure shows must
  also be readable from a table. Do not add a slide-deck builder, which was deliberately removed.

## Working in the repo

- Nothing lands on `main` except through a pull request; branch names are
  `<area>/<short-description>`, e.g. `anomaly/kgb-cia-conventions`.
- **Never run compute on a login node, and ask before submitting a Slurm job** — it spends shared
  allocation. Reading a TSV, `squeue`, `sacct` and importing a module are fine. See
  [`.agents/skills/anomaly/slurm-run/SKILL.md`](.agents/skills/anomaly/slurm-run/SKILL.md).
- **Generated data is read-only.** `config.refuse_overwrite` guards every stage output; re-run
  under a new `--out-prefix` or `--tag` rather than clearing a previous result out of the way.
- Large inputs and bulk outputs stay out of git: `*.npy`, `*.npz`, `out/edges.tsv`,
  `out/degrees.tsv`, `out/ranked_*.tsv` and the replication's per-edge CSV are gitignored. The
  small summaries in `results/` are committed so the reported numbers can be checked without a
  re-run.
- Skills live in `.agents/skills/<category>/<skill-name>/SKILL.md`. Edit those originals, never
  the generated links in `.claude/skills/`, and run `make skills` after adding or renaming one.
