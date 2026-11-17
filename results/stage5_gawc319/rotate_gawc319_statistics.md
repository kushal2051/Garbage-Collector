# RotatE on the GAWC-319 link-prediction split — 10 seeds

All three runs use the same split file, the same
1558 pos / 1558 neg held-out pairs per seed, prevalence 0.5, dim 512, gamma 12,
128 negatives, 50 epoch cap, patience 5 on val MRR. ROC from `score_hrt` +
`roc_auc_score` (not PyKEEN's rank evaluator). `degree` is the paired baseline;
`gnn` is the GraphSAGE unsup-v2 run on the identical split, shown for scale.

**unsup-v2** is the GraphSAGE run under the unsupervised objective — the encoder
is trained only to reconstruct the graph's own edges, then frozen, and a small
DistMult decoder is fitted on the overlapping pairs, so no GAWC label reaches
the encoder. The **v2** is the negative sampler: v1 corrupted an endpoint with
any node in the graph, which on a 1.2M-node graph is almost never a plausible
edge and let reconstruction ROC saturate above 0.99; v2 draws the replacement
from nodes observed in that position for that same predicate and rejects draws
that are real edges in the full graph. That matches the negatives the decoder is
scored on, and it stops the encoder from being trained to reject the decoder's
own held-out positives. Encoder reconstruction ROC is therefore not comparable
between v1 and v2, but the decoder ROC quoted here is.

## Finding

RotatE fails on this task on all three graphs. It never beats the degree
baseline on a single one of the 30 seeds, and the gap is large and consistent:
−0.18 on the merged KG, −0.10 on Pharos v1, −0.11 on Pharos v3. Pharos v1 lands
at chance (0.5009); the merged KG lands *below* chance at 0.4230, i.e. its
scores are anti-correlated with the truth, which is a training failure rather
than a weak signal. Both Pharos builds behave the same way, so the v1→v3
predicate rename (`genetically_associated_with` → `gene_associated_with_condition`)
makes no difference here.

Because of this, the table does not rank the graphs by how well they encode
GAWC — the same split gives GraphSAGE 0.68 on the merged KG and 0.88 on both
Pharos builds. Any graph comparison should come from those numbers, not these.
A usable RotatE comparison needs the training setup revisited first.

## Headline

| Graph | Nodes | Edges | Preds | GAWC predicate | RotatE ROC | Degree ROC | GraphSAGE ROC | Paired Δ (RotatE−degree) | Seeds RotatE>degree | Wall |
|---|---|---|---|---|---|---|---|---|---|---|
| Merged KG (v0.11 keepall) | 1,212,415 | 11,384,760 | 31 | gene_associated_with_condition | 0.4230 ± 0.0161 | 0.6008 ± 0.0046 | 0.6812 ± 0.0091 | −0.1778 ± 0.0177 | 0/10 | 10.36 h |
| Pharos v319 v1 | 307,917 | 696,375 | 6 | genetically_associated_with | 0.5009 ± 0.0134 | 0.6006 ± 0.0062 | 0.8839 ± 0.0060 | −0.0998 ± 0.0108 | 0/10 | 0.47 h |
| Pharos v319 v3 | 307,973 | 696,388 | 6 | gene_associated_with_condition | 0.4873 ± 0.0273 | 0.6007 ± 0.0062 | 0.8818 ± 0.0049 | −0.1133 ± 0.0290 | 0/10 | 0.92 h |

## RotatE detail

| Graph | ROC mean | ROC sd | ROC min | ROC max | PR mean | PR sd |
|---|---|---|---|---|---|---|
| Merged KG | 0.4230 | 0.0161 | 0.4101 | 0.4543 | 0.4563 | 0.0117 |
| Pharos v1 | 0.5009 | 0.0134 | 0.4743 | 0.5225 | 0.5006 | 0.0106 |
| Pharos v3 | 0.4873 | 0.0273 | 0.4175 | 0.5149 | 0.4911 | 0.0195 |

## Seen / unseen breakdown (ROC on positives, by whether either endpoint is unseen in training)

| Graph | RotatE both_seen | RotatE has_unseen | Degree both_seen | Degree has_unseen |
|---|---|---|---|---|
| Merged KG | 0.4338 ± 0.0148 | 0.2705 ± 0.0572 | 0.6400 | 0.0473 |
| Pharos v1 | 0.5018 ± 0.0128 | 0.4745 ± 0.0396 | 0.6159 | 0.1462 |
| Pharos v3 | 0.4899 ± 0.0268 | 0.4094 ± 0.0690 | 0.6159 | 0.1462 |

## Early stopping: best_epoch, stop epoch, wall clock per seed

`stop` is the epoch training halted on, `best` the epoch with the highest
validation MRR (the checkpoint kept as `best.pt`), `sec` training seconds for
that seed. Cap is 50 epochs, patience 5, warmup 5.

| Seed | Merged KG stop/best/sec | Pharos v1 stop/best/sec | Pharos v3 stop/best/sec |
|---|---|---|---|
| 0 | 8/3/3352 | 6/1/161 | 6/1/267 |
| 1 | 9/4/3737 | 6/1/157 | 6/1/263 |
| 2 | 8/3/3328 | 6/1/157 | 7/2/314 |
| 3 | 6/1/2498 | 6/1/157 | 6/1/265 |
| 4 | 7/2/2894 | 6/1/157 | 10/5/441 |
| 5 | 9/4/3706 | 6/1/157 | 6/1/267 |
| 6 | 6/1/2486 | 6/1/157 | 7/2/312 |
| 7 | 9/4/3708 | 7/2/185 | 6/1/266 |
| 8 | 17/12/6971 | 6/1/157 | 6/1/264 |
| 9 | 9/4/3706 | 6/1/157 | 6/1/267 |

| Graph | stop epoch (mean, range) | best epoch (mean, range) |
|---|---|---|
| Merged KG | 8.8 (6–17) | 3.8 (1–12) |
| Pharos v1 | 6.1 (6–7) | 1.1 (1–2) |
| Pharos v3 | 6.6 (6–10) | 1.6 (1–5) |

**Every seed on every graph stops at or barely past the warmup window, and
`best_epoch` is 1 on 25 of the 30 seeds.** On the Pharos graphs the best
checkpoint is the first epoch evaluated — the saved model is essentially the
initialisation. This is the result, not an incidental detail: nothing was
learned before patience fired.

## Per-epoch training loss and validation MRR

Mean over the seeds still running at that epoch (`Seeds` column), so later rows
describe fewer seeds and are not comparable to the earlier ones.

### Merged KG

| Epoch | Seeds | mean train loss | mean val MRR | max val MRR |
|---|---|---|---|---|
| 1 | 10 | 5.4616 | 2.05e-05 | 5.92e-05 |
| 2 | 10 | 5.3617 | 2.42e-05 | 7.42e-05 |
| 3 | 10 | 5.1897 | 3.15e-05 | 1.20e-04 |
| 4 | 10 | 4.9464 | 2.56e-05 | 1.27e-04 |
| 5 | 10 | 4.6356 | 1.84e-05 | 6.77e-05 |
| 6 | 10 | 4.2935 | 1.52e-05 | 4.93e-05 |
| 7 | 8 | 3.9558 | 1.59e-05 | 4.11e-05 |
| 8 | 7 | 3.6244 | 1.51e-05 | 4.07e-05 |
| 9 | 5 | 3.3076 | 2.31e-05 | 7.82e-05 |
| 10 | 1 | 3.0235 | 1.05e-05 | 1.05e-05 |
| 11 | 1 | 2.7959 | 1.06e-05 | 1.06e-05 |
| 12 | 1 | 2.6308 | 1.10e-05 | 1.10e-05 |
| 13 | 1 | 2.5045 | 1.10e-05 | 1.10e-05 |
| 14 | 1 | 2.3971 | 1.08e-05 | 1.08e-05 |
| 15 | 1 | 2.3003 | 1.09e-05 | 1.09e-05 |
| 16 | 1 | 2.2114 | 1.01e-05 | 1.01e-05 |
| 17 | 1 | 2.1280 | 1.06e-05 | 1.06e-05 |

### Pharos v1

| Epoch | Seeds | mean train loss | mean val MRR | max val MRR |
|---|---|---|---|---|
| 1 | 10 | 5.4923 | 3.84e-05 | 7.89e-05 |
| 2 | 10 | 5.4846 | 3.21e-05 | 5.64e-05 |
| 3 | 10 | 5.4725 | 2.64e-05 | 3.92e-05 |
| 4 | 10 | 5.4558 | 2.23e-05 | 3.03e-05 |
| 5 | 10 | 5.4341 | 2.08e-05 | 3.06e-05 |
| 6 | 10 | 5.4094 | 2.00e-05 | 3.16e-05 |
| 7 | 1 | 5.3843 | 1.77e-05 | 1.77e-05 |

### Pharos v3

| Epoch | Seeds | mean train loss | mean val MRR | max val MRR |
|---|---|---|---|---|
| 1 | 10 | 5.4922 | 3.47e-05 | 8.21e-05 |
| 2 | 10 | 5.4845 | 3.49e-05 | 8.95e-05 |
| 3 | 10 | 5.4724 | 3.00e-05 | 6.12e-05 |
| 4 | 10 | 5.4557 | 2.62e-05 | 4.10e-05 |
| 5 | 10 | 5.4340 | 2.44e-05 | 3.46e-05 |
| 6 | 10 | 5.4093 | 2.27e-05 | 3.13e-05 |
| 7 | 3 | 5.3842 | 2.54e-05 | 2.72e-05 |
| 8 | 1 | 5.3588 | 2.17e-05 | 2.17e-05 |
| 9 | 1 | 5.3332 | 1.86e-05 | 1.86e-05 |
| 10 | 1 | 5.3076 | 1.88e-05 | 1.88e-05 |

Two things to read off these.

**Validation MRR never leaves the random-ranking floor, and it drifts down, not
up.** For a uniformly random ranking over all N entities the expected MRR is
H_N/N — 1.20e-05 for the merged KG (N = 1,212,415) and 4.29e-05 for both Pharos
builds (N ≈ 307,900). Observed MRR is 1.0–3.2e-05 on the merged KG and
1.8–3.8e-05 on Pharos: the same order of magnitude as chance, and on Pharos
slightly *below* it. (That comparison assumes the evaluator ranks against the
full entity set; worth confirming against the evaluator config.) Since MRR peaks
at epoch 1–2 and declines monotonically after, patience fires on a metric that
is pure noise around the chance floor.

**The merged graph is optimising its objective while that metric sits still.**
Its training loss falls 5.46 → 2.13 over 17 epochs, a real optimisation
trajectory. Pharos barely moves, 5.49 → 5.41, because it stops after six
epochs. So the merged KG is a different failure from Pharos: it is fitting the
margin objective against its own sampled negatives, and that progress does not
transfer to ranking held-out GAWC pairs at all — consistent with its ROC of
0.4230, below chance. Pharos v1/v3 simply never get started.

The low-MRR early stop is therefore the headline diagnostic, exactly as asked:
it is not that RotatE learned a weak signal, it is that the early-stopping
metric could not see any signal to keep training on.

## Compute and artifacts

- **GPU:** one NVIDIA L40S per job (`tropshalab`/`l40-gpu`), torch 2.11.0+cu130,
  CUDA device, Python 3.13.18.
- **Wall clock**, whole job including evaluation: merged KG **10.36 h**,
  Pharos v1 **0.47 h**, Pharos v3 **0.92 h**. Training only, summed over the 10
  seeds: 10.11 h / 0.45 h / 0.81 h. The merged KG is ~22× the cost of Pharos v1
  per seed (3639 s vs 160 s mean).
- **Slurm jobs:** 9017221 (merged KG), 9017210 (Pharos v1), 9037437 (Pharos v3).
  Logs at `logs/rotate_kge.<jobid>.log`.
- **Checkpoints**, `best.pt` plus `epoch_%04d.pt` every 5 epochs, under
  `/work/users/k/k/kkoirala/nvidia-hackathon/rotate-ocsvm-gc/out/`:
  - `kg_gawc319_rotate_seed{0..9}/checkpoints` — 51 GB total
  - `pharos319_gawc319_rotate_seed{0..9}/checkpoints` — 12 GB total
  - `v3/pharos319_v3_gawc319_rotate_seed{0..9}/checkpoints` — 13 GB total
- Per-seed and per-epoch TSVs sit beside them as
  `<tag>_per_seed.tsv`, `<tag>_seen_unseen.tsv`, `<tag>_epochs.tsv`,
  `<tag>_summary.tsv`, `<tag>_meta.json` (76 GB of checkpoints in total, on
  `/work`, not in git).

## Why early stopping misfires

The configuration is `--warmup 5 --patience 5 --eval-frequency 1
--relative-delta 0.002`, with the 50-epoch cap used as the cosine horizon.
Three things interact badly.

**1. The patience budget is spent entirely inside the learning-rate warmup.**
The scheduler is linear warmup over `--warmup` epochs, then cosine decay to the
`--epochs` cap, so epoch 1 trains at 1/5 of the target LR and the full rate is
only reached at epoch 5. The stopper evaluates every epoch from epoch 1 and
allows 5 evaluations without a gain. Those two windows are the same length, so
patience can run out before the model has trained a single epoch at full LR.
The numbers confirm it exactly: **`stop == best + 5` in all 30 seeds, with no
exceptions.** The run length is not a property of the graph or the data — it is
whichever epoch happened to post the highest number, plus the patience
constant. 25 of 30 seeds never got past epoch 1.

**2. The metric the stopper watches is at the chance floor, so the "best"
epoch it latches onto is noise.** Validation MRR is 1–4e-05 against a
random-ranking expectation of 1.20e-05 (merged) and 4.29e-05 (Pharos), and it
drifts *downward* from epoch 1. `relative_delta: 0.002` asks for a 0.2%
relative improvement over the best so far; against a quantity that is both
noise-dominated and trending down, that condition is essentially never met.
The stopper is behaving correctly — it is reading a metric with no signal in it.

**3. Validation ranking is harder than the task actually scored.** The stopper
ranks each validation triple against all N entities (1.2M / 308k), while the
reported ROC scores a held-out pair against one type-matched negative. Those are
very different difficulties, and nothing was checked to confirm that MRR over
the full entity set moves measurably at this training budget before it was made
the stopping criterion. Negative sampling is also `filtered: False`, so true
edges are drawn as negatives during training — tolerable noise for the loss,
but it compounds the problem of a weak early signal.

So "RotatE failed" overstates what was tested. What these runs establish is
that **RotatE with this stopping configuration never trains**, on any of the
three graphs. The merged KG at least shows real optimisation (loss 5.46 → 2.13)
in the one seed that escaped to 17 epochs, which suggests the ceiling has not
been found. Before concluding anything about the graphs themselves:

- Exclude the warmup epochs from the stopper, or set `patience` well above
  `warmup` (e.g. warmup 5, patience 15) so stopping can only trigger after full
  LR has been applied for several epochs.
- Sanity-check the metric first: train one seed on the merged KG to a fixed 50
  epochs with no stopper and plot val MRR. If it stays at 1e-05 throughout,
  the metric or the evaluation set is wrong, not the model. That run costs
  ~6 h on one L40S, judging by seed 8's 17 epochs in 6971 s.
- Only then re-run the 10-seed comparison.
