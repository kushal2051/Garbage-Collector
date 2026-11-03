---
name: slurm-run
description: Submit and monitor this repo's stages on the Longleaf Slurm cluster. Use before running any pipeline stage, when choosing memory or walltime, when a job fails or is pending, when reading a job log, or when tempted to run a script directly in the shell.
---

# Running stages

**Nothing non-trivial runs on a login node.** Every stage goes through `sbatch`,
from the repo root so that `--output=logs/...` resolves:

```bash
sbatch slurm/extract.sbatch
sbatch slurm/degrees.sbatch
sbatch slurm/detect.sbatch biolink:treats 0.05
sbatch slurm/evaluate.sbatch --seeds 10
sbatch slurm/replicate.sbatch --kernels rbf --tag _rbf
sbatch slurm/plot_rbf.sbatch original_treats_rbf
```

**Ask the user before submitting.** Submitting a job spends shared allocation
and is not yours to decide. Propose the exact command and wait.

What *is* fine in the shell: `python -s -c "import config, rotate"`, reading a
TSV summary, `sacct`, `squeue`, inspecting a log. The line is compute, not
convenience.

## Site configuration

`slurm/env.sh` holds everything site-specific and is sourced by every `.sbatch`.
Override by exporting before `sbatch`, or with a flag:

| | |
|---|---|
| `PYTHON` | main interpreter; needs numpy, pandas, scipy, scikit-learn, matplotlib |
| `PYTHON_UMAP` | only for `replicate_original.py`, which needs `umap-learn` |
| `ROTATE_EMB` / `ROTATE_GRAPH` / `ROTATE_DEG_SRC` / `ROTATE_OUT` | inputs, consumed by `config.py` |
| partition | `sbatch -p <name> slurm/<stage>.sbatch` — the flag wins; Slurm does **not** expand shell variables in `#SBATCH` lines |

Two interpreters exist because `umap-learn` is absent from the main env. That is
a packaging accident, not a design choice: if you build one env with
`umap-learn`, point both at it.

Stages are invoked `$PYTHON -s script.py`. `-s` ignores user site-packages so a
stray `~/.local` install cannot shadow the env. **Not `-I`** — isolated mode
also drops the script's directory from `sys.path`, which breaks
`import config` / `import rotate`.

## Resources

Defaults in each `.sbatch`, sized from observed runs:

| stage | cpus | mem | time | note |
|---|---|---|---|---|
| extract | 4 | 16G | 40m | one streaming pass, I/O bound |
| degrees | 4 | 16G | 40m | two counters over a 1 GB TSV |
| detect | 4 | 24G | 1h | SVM on ~9.7k × 1024; `node_embeddings.npy` is mmapped, so RSS stays well under its 1.9 GB |
| evaluate | 8 | 48G | 8h | 10 seeds × nu grid of 9; the expensive one |
| replicate | 8 | 32G | 4h | UMAP dominates; the 32-point grid is the original's |
| plot_rbf | 4 | 24G | 1h | refits one SVM to recover the decision function |

A OneClassSVM on this data is O(n²) in the edge count. A predicate with far more
than ~10k edges needs its walltime raised before submitting, not after it times
out.

## Reading results

Logs land in `logs/<stage>.<jobid>.log` (stdout and stderr together). Each
stage prints its input counts first and its output paths last, so a truncated
log still says how far it got.

Two failures are expected and are not bugs:

- *missing input(s) … Set ROTATE_EMB / …* — an earlier stage has not run, or an
  input moved. Run the earlier stage; do not reach around `config.py`.
- *refusing to overwrite …* — generated data is read-only. Pass a different
  `--out-prefix` (evaluate) or `--tag` (replicate), or move the old result
  aside. Both checks run in the first second, before any compute.
