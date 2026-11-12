"""Stage 5d. RotatE link prediction, scored on exactly the rows stage 5c was scored on.

Why a second method at all (KGB-CIA issue #78): stage 5c's unsupervised encoder trains and
early-stops on link reconstruction against UNIFORMLY random negatives, which saturates above
0.99 within 400 steps on both graphs. After that neither the metric nor the loss can move, so
the merged-KG encoder stops having seen a fraction of its 11.4M edges once, and its decoder
lands one point above the degree baseline. Whether the Pharos-vs-merged gap is the graphs'
content or the encoder's cannot be read off that run.

RotatE does not share that weakness. It scores the relation directly -- score(s, r, o) =
gamma - ||h o r - t|| with r a unit-modulus rotation -- so there is no decoder, and its
self-adversarial (NSSA) loss weights hard negatives, so the objective does not go flat once
easy negatives are separated.

What makes the two numbers comparable is that NOTHING about the protocol changes: the same
split file, the same held-out pairs, the same negatives, the same training edges, the same
degree baseline, the same seen/unseen mask -- all of it from `linkpred.Comparison`, the same
object gnn.py uses. The model is the only thing that differs.

Two of the four models reported are model-free and are the check that the pairing held: the
`degree` ROC and the per-subset `n_pos` MUST come out identical to the stage 5c run on the same
split and seeds. If they do not, the comparison is void regardless of what RotatE scored.

Hyperparameters are the ones from the earlier RotatE jobs, as issue #78 specifies: 512 real
dimensions (256 complex), gamma 12, adversarial temperature 1, 128 negatives per positive,
batch 8192, Adam 1e-3 with 5 warm-up epochs then cosine decay, unfiltered negatives, no weight
decay. 50 epochs is a CAP; early stopping on validation MRR decides when to stop.

Usage:
    python rotate_kge.py --graph kg --predicate biolink:gene_associated_with_condition \
           --split out/split_gawc319_pairs.tsv --tag kg_gawc319_rotate --seeds 1
    python rotate_kge.py --graph pharos319 --predicate biolink:genetically_associated_with \
           --split out/split_gawc319_pairs.tsv --tag pharos319_gawc319_rotate --seeds 1
"""

import argparse
import math
import platform
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.training.callbacks import TrainingCallback
from pykeen.triples import TriplesFactory

import config
import kgraph
import linkpred
from linkpred import MIN_SUBSET

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="kg", help="a tag prepared by kg_prepare.py, or a .npz path")
ap.add_argument("--predicate", default="biolink:gene_associated_with_condition")
ap.add_argument("--split", default=None, help="a pairs.tsv from pharos_split.py")
ap.add_argument("--seeds", type=int, default=1, help="split seeds, 0..n-1, as in gnn.py")
ap.add_argument("--epochs", type=int, default=50, help="a CAP; the stopper decides")
ap.add_argument("--warmup", type=int, default=5, help="epochs of linear warm-up before cosine")
ap.add_argument("--batch", type=int, default=8192)
ap.add_argument("--negatives", type=int, default=128, help="negatives per positive")
ap.add_argument("--dim", type=int, default=512, help="REAL dimensions; RotatE uses dim/2 complex")
ap.add_argument("--gamma", type=float, default=12.0, help="NSSA margin, and the init range")
ap.add_argument("--adversarial-temperature", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=5, help="evaluations without a validation gain")
ap.add_argument(
    "--relative-delta",
    type=float,
    default=0.002,
    help="PyKEEN's default 0.01 stops on the small late gains cosine decay produces",
)
ap.add_argument("--eval-frequency", type=int, default=1, help="epochs between validations")
ap.add_argument("--checkpoint-every", type=int, default=5)
ap.add_argument("--model-seed", type=int, default=29, help="RotatE's init and sampling seed")
ap.add_argument("--train-frac", type=float, default=0.8)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--tag", default="rotate", help="output prefix")
ap.add_argument("--no-write", action="store_true")
args = ap.parse_args()

DEVICE = torch.device(args.device)
COMPLEX_DIM = args.dim // 2

npz = config.OUT / f"{args.graph}.npz" if not args.graph.endswith(".npz") else args.graph
config.require(npz)
outputs = linkpred.output_paths(args.tag)
EPOCH_TSV = config.OUT / f"{args.tag}_epochs.tsv"
if not args.no_write:
    config.refuse_overwrite(*outputs, EPOCH_TSV)

graph = kgraph.load(npz)
print(graph.describe(), flush=True)

C = linkpred.Comparison(
    graph,
    predicate=args.predicate,
    split=args.split,
    train_frac=args.train_frac,
    val_frac=args.val_frac,
)

# Entity ids are the graph's OWN node ids, over every node of the node table and not just the
# ones with a training edge: a test endpoint with no training edge must still get an (untrained)
# embedding rather than a KeyError, which is what the has_unseen rows measure. Relation ids are
# the graph's predicate list. Both are already the space `Comparison` returns triples in, so no
# triple is ever relabelled between the protocol and the model.
ENTITY_TO_ID = C.index
RELATION_TO_ID = {predicate: i for i, predicate in enumerate(graph.predicates)}


def factory(triples: np.ndarray) -> TriplesFactory:
    return TriplesFactory(
        mapped_triples=torch.as_tensor(np.ascontiguousarray(triples), dtype=torch.long),
        entity_to_id=ENTITY_TO_ID,
        relation_to_id=RELATION_TO_ID,
        create_inverse_triples=False,
        num_entities=graph.num_nodes,
        num_relations=graph.num_predicates,
    )


class SnapshotEvery(TrainingCallback):
    """A checkpoint every k epochs, so a stopped run is still inspectable."""

    def __init__(self, every: int, out) -> None:
        super().__init__()
        self.every, self.out = every, out

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs) -> None:
        if self.every and epoch % self.every == 0:
            torch.save(self.training_loop.model.state_dict(), self.out / f"epoch_{epoch:04d}.pt")


def learning_rate(epoch: int) -> float:
    """Linear warm-up, then cosine decay over the remaining epochs of the cap."""
    if epoch < args.warmup:
        return (epoch + 1) / args.warmup
    return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))


class WarmupCosine(torch.optim.lr_scheduler.LambdaLR):
    """The schedule above, baked into a class.

    PyKEEN rebuilds the scheduler before training from `state_dict()` filtered by `__init__`'s
    signature, and a LambdaLR's `lr_lambdas` does not survive that round trip -- passing
    `lr_scheduler="LambdaLR"` with an `lr_lambda` kwarg raises `missing 1 required positional
    argument` inside the training loop. A subclass that takes only the optimizer reconstructs
    correctly.
    """

    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer, lr_lambda=learning_rate)


def run_seed(seed: int):
    """Train RotatE on this seed's training edges; score this seed's test rows."""
    parts, _ = C.positives_and_training_edges(seed)
    val_pos, test_pos = parts["val"], parts["test"]
    # The training pairs need no separate fitting stage: unlike the GraphSAGE decoder,
    # RotatE sees them as ordinary triples of `train_edges` below.
    if len(test_pos) < MIN_SUBSET:
        raise SystemExit(f"seed {seed}: only {len(test_pos)} test positives")

    # The stricter hold-out: every held-out pair gone under ANY predicate and in EITHER
    # direction. RotatE trains on the triples themselves, so a test pair left in under some
    # other predicate is the pair it is later scored on. Identical to the stage 5c encoder's
    # edge set, which is what makes the degree baseline below comparable.
    train_edges = C.encoder_training_edges(parts)
    print(
        f"  RotatE trains on {len(train_edges):,} of {graph.num_edges:,} triples "
        f"({graph.num_edges - len(train_edges):,} held out), {len(val_pos):,} validation pairs, "
        f"{len(test_pos):,} test pairs",
        flush=True,
    )

    run_dir = config.OUT / f"{args.tag}_seed{seed}"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    emb_range = (args.gamma + 2.0) / COMPLEX_DIM  # Sun et al.'s init, 14/256 at the defaults
    started = time.time()
    result = pipeline(
        training=factory(train_edges),
        validation=factory(val_pos),
        testing=factory(val_pos),  # required by the pipeline; the test rows are scored below
        model="RotatE",
        model_kwargs={
            "embedding_dim": COMPLEX_DIM,
            "entity_initializer": partial(torch.nn.init.uniform_, a=-emb_range, b=emb_range),
        },
        loss="NSSA",
        loss_kwargs={
            "margin": args.gamma,
            "adversarial_temperature": args.adversarial_temperature,
        },
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": args.negatives, "filtered": False},
        optimizer="Adam",
        optimizer_kwargs={"lr": args.lr, "weight_decay": 0.0},
        lr_scheduler=WarmupCosine,
        training_kwargs={
            "num_epochs": args.epochs,
            "batch_size": args.batch,
            "callbacks": SnapshotEvery(args.checkpoint_every, checkpoints),
        },
        stopper="early",
        stopper_kwargs={
            "frequency": args.eval_frequency,
            "patience": args.patience,
            # MRR, not hits@10: hits@10 over 1,557 validation pairs moves in steps of 1/1,557
            # and the stopper cannot see a gain smaller than one pair crossing the threshold.
            "metric": "inverse_harmonic_mean_rank",
            "relative_delta": args.relative_delta,
            "best_model_path": checkpoints / "best.pt",
            "clean_up_checkpoint": False,
        },
        random_seed=args.model_seed,
        device=args.device,
    )
    seconds = time.time() - started
    if not args.no_write:
        result.save_to_directory(str(run_dir))

    # The stopper reloads the best weights when it stops; score the held-out rows with them.
    model = result.model.eval()
    relation = RELATION_TO_ID[args.predicate]
    test_neg = C.negatives_for(test_pos, seed * 7 + 2)

    def score(triples: np.ndarray) -> np.ndarray:
        hrt = torch.as_tensor(np.ascontiguousarray(triples), dtype=torch.long)
        hrt[:, 1] = relation
        with torch.no_grad():
            return model.score_hrt(hrt.to(model.device)).squeeze(-1).cpu().numpy()

    scores = {
        "rotate_pos": score(test_pos),
        "rotate_neg": score(test_neg),
        "degree_pos": kgraph.degree_scores(test_pos, train_edges, graph.num_nodes),
        "degree_neg": kgraph.degree_scores(test_neg, train_edges, graph.num_nodes),
    }

    stopper = result.stopper
    best_epoch = getattr(stopper, "best_epoch", None)
    best_metric = getattr(stopper, "best_metric", None)
    # `stopper.results` holds one validation MRR per evaluation, which happens every
    # `--eval-frequency` epochs; `result.losses` holds one training loss per epoch.
    validations = list(getattr(stopper, "results", []) or [])
    history = []
    for index, loss in enumerate(result.losses):
        epoch = index + 1
        evaluation = epoch // args.eval_frequency - 1 if epoch % args.eval_frequency == 0 else -1
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(loss),
                "val_mrr": (
                    float(validations[evaluation]) if 0 <= evaluation < len(validations) else None
                ),
            }
        )
    print(
        f"  stopped at epoch {len(result.losses)} (best {best_epoch}), val MRR "
        f"{best_metric if best_metric is None else round(float(best_metric), 4)}, "
        f"{seconds:.0f}s, checkpoints in {checkpoints}",
        flush=True,
    )
    return scores, kgraph.both_seen(test_pos, train_edges), best_metric, history, seconds


rows, seen_rows, epochs = [], [], []
started = time.time()
for seed in range(args.seeds):
    scores, mask, best_val, history, seconds = run_seed(seed)
    for name in ("rotate", "degree"):
        row, seen = linkpred.score_rows(seed, name, scores, mask)
        rows.append(row)
        seen_rows.extend(seen)
    epochs.extend(history)
    print(f"seed {seed} done ({time.time() - started:.0f}s total)", flush=True)

per_seed, summary, prevalence = linkpred.summarise(rows)
seen_unseen = pd.DataFrame(seen_rows)
linkpred.report(
    f"{args.graph}, {args.predicate}, RotatE, {args.seeds} split seed(s)",
    summary,
    prevalence,
    seen_unseen,
)
print(
    "\nThe `degree` row and the per-subset n_pos must match the GraphSAGE run on this split and "
    "these seeds exactly; they are model-free, so a difference means the pairing broke."
)

if not args.no_write:
    pd.DataFrame(epochs).to_csv(EPOCH_TSV, sep="\t", index=False)
    linkpred.write(
        outputs,
        per_seed,
        summary,
        seen_unseen,
        {
            "args": vars(args),
            "graph": {
                "tag": args.graph,
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "num_predicates": graph.num_predicates,
                "dropped": graph.dropped,
            },
            "prevalence": prevalence,
            "models": ["rotate", "degree"],
            "torch": torch.__version__,
            "device": str(DEVICE),
            "python": platform.python_version(),
            "seconds": round(time.time() - started, 1),
            "note": "RotatE under the stage-5 protocol: same split file, same held-out pairs, "
            "same negatives and same training edges as the GraphSAGE run, so the two ROC "
            "numbers are paired row for row. Scored with score_hrt and roc_auc_score, not "
            "PyKEEN's rank-based evaluator.",
        },
    )
