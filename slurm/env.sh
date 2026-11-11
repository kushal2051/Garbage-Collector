# Sourced by every .sbatch in this directory. Everything site-specific lives
# here, so moving the pipeline to another cluster is one file.
#
# Override any of these in your shell before calling sbatch, e.g.
#   SLURM_PARTITION=general sbatch slurm/detect.sbatch

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The stages are invoked as `$PYTHON -s script.py`. -s ignores the user
# site-packages dir, so a stray ~/.local install cannot shadow the env. NOT -I:
# isolated mode also drops the script's directory from sys.path, which would
# break `import config` / `import rotate`.
#
# Interpreter. Needs numpy, pandas, scipy, scikit-learn, matplotlib.
# The replication stage additionally needs umap-learn -- see PYTHON_UMAP.
PYTHON="${PYTHON:-/work/users/k/k/kkoirala/envs/pure/bin/python}"

# umap-learn is not installed in the env above, which is why the replication
# stage uses a different one. This is a packaging accident, not a design choice:
# if you build a single env with umap-learn, point both at it.
PYTHON_UMAP="${PYTHON_UMAP:-/work/users/k/k/kkoirala/envs/tmaps/bin/python}"

# The link-prediction stages (kg_prepare, pharos_split, gnn) additionally need torch. Same
# packaging accident as umap above: torch is not in the repo's dependencies, because only these
# stages use it and it has to match the cluster's CUDA build. kg_prepare and pharos_split are
# numpy-only and run under $PYTHON too; gnn.py needs this one.
PYTHON_GNN="${PYTHON_GNN:-/work/users/k/k/kkoirala/envs/esmc_sweep/bin/python}"

LOGS="${LOGS:-$REPO/logs}"
mkdir -p "$LOGS"

# Input locations, consumed by config.py. Defaults there match these.
export ROTATE_EMB="${ROTATE_EMB:-/work/users/k/k/kkoirala/nvidia-hackathon/rotate_original_embeddings}"
export ROTATE_GRAPH="${ROTATE_GRAPH:-/work/users/k/k/kkoirala/n2v2sae/graphs/robokop_30fd_filtered}"
export ROTATE_DEG_SRC="${ROTATE_DEG_SRC:-/work/users/k/k/kkoirala/CyclopsHGT/data_set/30fd_robo/edges.tsv}"
export ROTATE_OUT="${ROTATE_OUT:-$REPO/out}"

# The merged KGB-CIA graph and the Pharos baseline, consumed by config.py.
export ROTATE_KG="${ROTATE_KG:-/work/users/k/k/kkoirala/nvidia-hackathon/merged}"
export ROTATE_PHAROS="${ROTATE_PHAROS:-/work/users/k/k/kkoirala/UNC8-dtd-project/UNC8/targets/data/01_raw/pharos_edges.jsonl}"

cd "$REPO"
