# `experimental/` — crystal-graph GNN for lattice parameters (OLCF Frontier)

> **Status (2026-08-13): EXPERIMENTAL. Never run on Frontier. No accuracy claim exists.**
> Validated on CPU only. Nothing here is imported by `cli.py`, `server.py`, or `engine/`, and
> nothing here is covered by `tests/`. Do not cite a number from it.

This directory is deliberately outside `engine/`: `engine/` is production code with test
coverage and a stable interface, and this is not that yet.

## What this adds to the existing pipeline

The shipped models (`engine/train_models.py`) predict `[a, b, c]` from **composition +
symmetry** — 503 matminer features, crystal system, spacegroup, nsites. No geometry ever
reaches them, because `config.QUERY_FIELDS` treats the MP `structure` as transient.

This adds a **geometry-aware** alternative: a CGCNN-style graph network over the crystal
structure itself.

| | RF/GBR baseline | this GNN |
|---|---|---|
| Input | composition + symmetry scalars | crystal graph + symmetry scalars |
| Target | `[a, b, c]` | `[a, b, c]` (same) |
| Filters | dedup + `a,b,c ≤ 10 Å` | **identical** (parity is deliberate) |
| Needs a structure at inference? | no | **yes** — see caveat below |

## Files

| File | Role |
|---|---|
| `graph_dataset.py` | Re-queries MP, builds and caches PyG graphs into `Dataset/graphs/` (gitignored). Applies the same dedup + ≤10 Å filters as `prepare_training_frame()`. |
| `train_lattice_gnn.py` | Multi-node DDP trainer. `CGConv` stack + mean pooling + symmetry descriptors → `[a, b, c]`. Reports MAE and **MAE_cubic** so it lines up with the baselines. |
| `submit_lattice_gnn.sh` | Frontier SLURM script, 2 nodes / 16 GCDs / 30 min. Site config is placeholdered — substitute into a working copy, never commit real values. |

`train_lattice_gnn.py` imports **only** torch / PyG / mpi4py — nothing from `engine/`. That is
intentional: the trainer is the artifact that ships to a supercomputer, so it must not drag
pymatgen, mp_api, or the rest of this repo onto Frontier with it.

## Read this before trusting any MAE: target leakage

A crystal graph built from a relaxed structure **contains the answer** — interatomic
distances *are* the lattice. A network fed raw distances reads `a, b, c` straight off its
edge features and posts a near-zero MAE that means nothing, while the RF baseline it is
being compared against never sees geometry at all.

`graph_dataset.py` therefore has two edge modes:

- **`--edge-scale free` (default, the real task).** Distances are divided by each
  structure's own mean nearest-neighbour distance before Gaussian expansion. Edge features
  carry *shape and coordination*, not size. The model must infer absolute scale from species
  identity and local environment. **Only these runs are comparable to RF/GBR.**
- **`--edge-scale raw` (diagnostic only).** Real ångström distances, which the network can in
  principle integrate back into the cell dimensions. **Never report it as a result.**

  > **The leakage check is currently INCONCLUSIVE.** Run on the smoke subset (188 training
  > graphs, 8 epochs, CPU), `raw` reached val MAE **1.9671 Å** against `free`'s **1.6678 Å** —
  > the two modes did not separate, and `raw` was not better. That is a budget artefact, not
  > evidence that `raw` is safe: neither mode is anywhere near converged. The comparison has
  > to be repeated at a real training budget before anyone concludes how much `raw` actually
  > leaks. `free` remains the default because it is leak-free *by construction*, which does
  > not depend on this experiment coming out any particular way.

`train_lattice_gnn.py` reads the dataset's `.meta.json` and prints a loud warning when it is
handed a `raw` set.

## Inference caveat (a real limitation, not a detail)

The RF pipeline can predict a novel solid solution — `UN0.5C0.5` has no MP entry, and it
doesn't need one, because composition plus a curated reference symmetry is enough. **This
GNN needs a structure.** For a novel solid solution you must supply a **template**: take the
reference host structure (`Data/reference_systems.json` / `mp_client.reference_symmetry`) and
substitute species onto its sites. Scale-free edge features are what make that legitimate —
the template's own lattice constant is normalised away — but the *prototype* assumption
remains and must be stated in any result. If that assumption is unacceptable, the
composition-graph (Roost/CrabNet-style) formulation is the alternative, and needs no
structure at all.

## Running it

**1. Build graphs where there is internet.** OLCF **compute nodes have no network access**,
so the MP query can never run inside the job. Build on a workstation or an OLCF login node
and transfer the `.pt`.

```bash
# U-only smoke subset (400 docs -> 208 graphs)
python -m experimental.graph_dataset --elements U --limit 400 --tag Usubset

# full U-only build (2,458 docs -> 917 graphs) — this is what has been exercised
python -m experimental.graph_dataset --elements U --tag Uall

# whole database — hours, and tens of GB of structures over the wire. Never run.
python -m experimental.graph_dataset --tag all
```

Needs `MP_API_KEY` in `.env`. Output lands in `$LATTICE_DATA_ROOT/Dataset/graphs/` (or
`Dataset/graphs/` when `LATTICE_DATA_ROOT` is unset) — **outside this repo if that variable
points elsewhere**, which is easy to miss when writing a transfer step.

**2. Transfer, then train.**

```bash
rsync -avh graphs_Uall_free.pt <user>@dtn.olcf.ornl.gov:<dest>/graphs/
sbatch -A <account> experimental/submit_lattice_gnn.sh
GRAPHS=/path/to/graphs_Uall_free.pt sbatch -A <account> experimental/submit_lattice_gnn.sh
```

**Load-check the `.pt` on a login node before queueing.** It is a pickle of PyG `Data`
objects (`torch.load(..., weights_only=False)`), not a tensor blob, so it needs a compatible
`torch_geometric` on the far side. Built here against torch 2.12.1 / PyG 2.8.0; the Frontier
env targets torch 2.10.0 (ROCm 7.1). Cross-version load is **unverified**.

The env needs `torch` (ROCm build), `torch_geometric` **with the `-rocm` companion wheels**,
and `mpi4py` built against Cray MPICH. Verified against the OLCF doc 2026-08-13:

```bash
pip install ninja packaging scipy
pip install torch-geometric torch-sparse-rocm torch-spline-conv-rocm \
            torch-scatter-rocm torch-cluster-rocm pyg-lib-rocm
```

## Baselines, and what would count as beating them

From `Results/metrics/ModelMetrics_CrossVal.csv` (quoted in `engine/config.py`), **5-fold CV,
cubic subset**:

- `gbr1` Lumped GBR (XGBoost) — MAE_cubic **0.113556 Å** (best CV)
- `rf1` Lumped RF — MAE_cubic **0.121701 Å** (headline model)

`train_lattice_gnn.py` currently reports a **holdout** split, which is *not* the same
protocol. To make a real claim, port the model into `engine/evaluate_cv.py`'s 5-fold
protocol and run both benchmarks (`Results/benchmarks/basis_check.md`) — and remember the
warning already in `config.py`: benchmark MAE is contaminated by the DFT-vs-experiment
label-basis offset, so compare with **Pearson r / slope**, not absolute MAE.

## Known limitations at full scale (found in review, not yet hit)

The largest build exercised is 917 graphs. Three things do not obviously survive a full
~150k-structure build, and should be dealt with **before** any full run:

1. **Pass 1 holds every conventional `Structure` in a pandas column.** `build_graphs()`
   collects metadata *and* the pymatgen object for each material so dedup/filtering can
   happen before the expensive graph step. At 150k structures that is a large resident
   footprint. Fix if it bites: chunk the query, or persist structures to disk and re-read.
2. **Output is a single monolithic `.pt`.** `torch.save(graphs, path)` writes one file and
   `torch.load` reads it whole. Shard it (or move to PyG's `InMemoryDataset`/on-disk
   `Dataset`) once a full build exists.
3. **Every rank loads the entire dataset.** `train_lattice_gnn.py` calls `torch.load` per
   rank, so a Frontier node running 8 tasks holds **8 copies** in RAM. Fine at 917 graphs,
   potentially not for the full set — shard by rank at load time, or memory-map.

None are correctness bugs; all three are silent-until-large.

## What has actually been verified

- ✅ Graph builder runs against live MP. Full U-only build: **2,458** U-containing entries →
  2,457 after dedup → **917 graphs** after the ≤10 Å filter (34,060,029 bytes).
- ✅ Trainer runs end to end on CPU/gloo at **world_size 1, 2 and 4**; loss and val MAE both
  decrease, all ranks exit 0, and the cross-rank metric reduction is exercised.
- ✅ Validation sharding is **disjoint by construction** — plain striding, not
  `DistributedSampler`, which pads a split by repeating samples (20 items across 16 ranks
  becomes 32 sampled with 12 duplicates, i.e. 37.5 % of the metric double-counted).
- ✅ PyG `-rocm` wheel names verified verbatim against the OLCF doc (2026-08-13).
- ❌ **Never run on Frontier.** No ROCm, no multi-node, no RCCL path exercised. The `hsn0`
  and per-step-port gotchas are encoded but untested here.
- ❌ **No accuracy claim.** Holdout ≠ the baselines' 5-fold CV, and only 21 of the 91
  validation graphs in the U-only build are cubic.
- ❌ **Leakage check inconclusive** at smoke budget (see the callout above).
- ❌ **Cross-version `.pt` load unverified** (torch 2.12.1/PyG 2.8.0 → torch 2.10.0).
- ❌ **Template-substitution inference is designed but not implemented.** Nothing here yet
  predicts a novel solid solution the way `cli.py predict` does.
