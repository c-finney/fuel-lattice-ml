#train_lattice_gnn.py
# Multi-node DDP training of a CGCNN-style crystal-graph network that predicts [a, b, c]
# from a Materials Project structure — the GNN counterpart to the RF/GBR models in
# engine/train_models.py.
#
# FRONTIER BOILERPLATE (see skills/frontier-pytorch/SKILL.md in the fuel-agent repo):
#   - mpi4py imported before torch
#   - NCCL_SOCKET_IFNAME via setdefault('hsn0') — honors the batch-script export; hsn0 is
#     OLCF's documented remedy for multi-node hangs, their default is all four for
#     bandwidth at scale (deliberate deviation, see SKILL.md #1)
#   - env:// rendezvous; each srun step needs its OWN --master_port
#   - one GPU per task (--gpus-per-task=1) => local_rank is always 0
#
# BASELINES TO BEAT (5-fold CV, cubic subset, from Results/metrics/ModelMetrics_CrossVal.csv
# as quoted in engine/config.py):
#     gbr1 (Lumped GBR/XGBoost)  MAE_cubic = 0.113556 A   <- best CV
#     rf1  (Lumped RF)           MAE_cubic = 0.121701 A   <- headline model
# This script prints MAE over all systems AND over the cubic subset so the comparison is
# like-for-like. It is a HOLDOUT split, not 5-fold CV — see the note in main().
#
# >>> LEAKAGE <<< Only compare a run built with --edge-scale free (the default in
# experimental/graph_dataset.py) against those baselines. A "raw" dataset hands the model the
# answer through its edge features; it is a plumbing check, never a result.
#
# STATUS 2026-08-11: CPU-validated on a capped U-containing subset. NOT yet run on Frontier.
from mpi4py import MPI          # MUST precede torch

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool

MAX_Z = 100


class CrystalGNN(torch.nn.Module):
    """CGCNN-style: element embedding + continuous node features, edge-conditioned
    convolutions (CGConv), mean-pooled readout concatenated with graph-level descriptors
    (crystal-system one-hot, spacegroup, nsites), then an MLP to [a, b, c]."""

    def __init__(self, node_cont_dim: int, edge_dim: int, u_dim: int,
                 hidden: int = 128, num_layers: int = 4, out_dim: int = 3):
        super().__init__()
        self.embed = torch.nn.Embedding(MAX_Z, hidden)
        self.node_in = torch.nn.Linear(node_cont_dim, hidden)
        self.convs = torch.nn.ModuleList(
            [CGConv(hidden, dim=edge_dim, batch_norm=True) for _ in range(num_layers)]
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + u_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, data):
        h = self.embed(data.z) + self.node_in(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)
        h = global_mean_pool(h, data.batch)
        u = data.u.view(h.size(0), -1)
        return self.head(torch.cat([h, u], dim=1))


def evaluate(model, loader, device, cubic_only: bool = False):
    """Return (sum of absolute errors, count) so ranks can be reduced without bias."""
    model.eval()
    abs_err, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if cubic_only:
                keep = [i for i, cs in enumerate(batch.crystal_system) if cs == "cubic"]
                if not keep:
                    continue
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)
            y = batch.y.view_as(pred)
            per_graph = (pred - y).abs().mean(dim=1)
            if cubic_only:
                idx = torch.tensor(keep, device=device, dtype=torch.long)
                per_graph = per_graph[idx]
            abs_err += per_graph.sum().item()
            n += per_graph.numel()
    return abs_err, n


def main(args, local_rank, global_rank, world_size):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    graphs = torch.load(args.graphs, weights_only=False)
    meta_path = Path(str(args.graphs).replace(".pt", ".meta.json"))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if meta.get("edge_scale") == "raw" and global_rank == 0:
        print("!! WARNING: this dataset was built with edge_scale='raw', which LEAKS the "
              "target through edge distances. Any MAE below is a plumbing check, NOT a "
              "model result, and must not be compared to the RF/GBR baselines.", flush=True)

    # Deterministic holdout split — every rank must produce the SAME split, so seed it.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(graphs), generator=g).tolist()
    n_val = max(1, int(args.val_frac * len(graphs)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=args.num_workers,
        sampler=DistributedSampler(train_set, num_replicas=world_size, rank=global_rank),
    )
    # Validation is sharded by PLAIN STRIDING, not DistributedSampler. DistributedSampler
    # pads a split by repeating samples so every rank gets an equal count — on 20 val graphs
    # across 16 ranks that is 32 sampled with 12 duplicates, i.e. 37.5% of the reported
    # metric is double-counted. Striding gives each rank a disjoint slice, so the reduced
    # sum/count below is exact. Ranks may get different counts (or none); that is safe here
    # because DDP only collectives on backward, and evaluate() runs under no_grad.
    val_shard = val_set[global_rank::world_size]
    val_loader = DataLoader(val_shard, batch_size=args.batch_size, shuffle=False)

    sample = graphs[0]
    model = CrystalGNN(
        node_cont_dim=sample.x.size(1),
        edge_dim=sample.edge_attr.size(1),
        u_dim=sample.u.numel(),
        hidden=args.hidden,
        num_layers=args.num_layers,
    ).to(device)
    # First cross-node collective happens in this constructor. stdout stopping here means
    # rendezvous/interface, not the model. See SKILL.md gotcha #1.
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if global_rank == 0:
        print(f"world_size={world_size} device={device} graphs={len(graphs)} "
              f"train={len(train_set)} val={len(val_set)} "
              f"edge_scale={meta.get('edge_scale', 'unknown')}", flush=True)
        print(f"model: node_cont={sample.x.size(1)} edge_dim={sample.edge_attr.size(1)} "
              f"u_dim={sample.u.numel()} hidden={args.hidden} layers={args.num_layers}",
              flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loader.sampler.set_epoch(epoch)
        run_loss, n_seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            pred = model(batch)
            loss = F.smooth_l1_loss(pred, batch.y.view_as(pred), beta=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()         # <- the gradient all-reduce
            optimizer.step()
            run_loss += loss.item() * batch.num_graphs
            n_seen += batch.num_graphs
        sched.step()

        # Reduce SUMS and COUNTS (not per-rank means) so uneven shards don't skew the number.
        val_err, val_n = evaluate(model, val_loader, device)
        cub_err, cub_n = evaluate(model, val_loader, device, cubic_only=True)
        stats = torch.tensor([run_loss, n_seen, val_err, val_n, cub_err, cub_n],
                             device=device, dtype=torch.double)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        if global_rank == 0:
            train_loss = stats[0] / max(stats[1], 1)
            val_mae = stats[2] / max(stats[3], 1)
            cub_mae = stats[4] / max(stats[5], 1) if stats[5] > 0 else float("nan")
            print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val MAE {val_mae:.4f} A | "
                  f"val MAE_cubic {cub_mae:.4f} A (n={int(stats[5])}) | "
                  f"{time.time() - t0:.1f}s", flush=True)
            if args.out_dir and epoch % args.save_every == 0:
                out = Path(args.out_dir)
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"MODEL_STATE": model.module.state_dict(),
                            "EPOCH": epoch,
                            "meta": meta,
                            "val_mae": float(val_mae),
                            "val_mae_cubic": float(cub_mae)},
                           out / "lattice_gnn.pt")

    if global_rank == 0:
        print("\n--- comparison (5-fold CV numbers, this run is a holdout split) ---")
        print("  gbr1 Lumped GBR  MAE_cubic = 0.113556 A")
        print("  rf1  Lumped RF   MAE_cubic = 0.121701 A")
        print("  A holdout MAE is NOT a CV MAE. To claim a win, port this into "
              "engine/evaluate_cv.py's 5-fold protocol.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DDP crystal-graph GNN for lattice parameters")
    p.add_argument("--graphs", type=str, required=True,
                   help="path to a graphs_*.pt from experimental/graph_dataset.py")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64, help="per-rank batch size")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--master_addr", type=str, required=True)
    p.add_argument("--master_port", type=str, required=True)
    args = p.parse_args()

    num_gpus_per_node = max(torch.cuda.device_count(), 1)   # == 1 with --gpus-per-task=1

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    global_rank = comm.Get_rank()
    local_rank = global_rank % num_gpus_per_node            # always 0 with 1 GPU/task

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(args.master_addr)
    os.environ["MASTER_PORT"] = str(args.master_port)

    # Honor the batch-script export; default to hsn0 if unset. NEVER assign
    # os.environ[...] = 'hsn0,hsn1,hsn2,hsn3' — a bare assignment silently overrides the
    # batch script. On the default itself: OLCF recommends all four NICs for best
    # bandwidth at scale and gives hsn0-only as the documented remedy for multi-node
    # hangs (verified 2026-08-13). We apply the remedy up front — see SKILL.md #1.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")

    backend = "nccl" if torch.cuda.is_available() else "gloo"   # nccl == RCCL on ROCm/AMD
    dist.init_process_group(backend=backend, init_method="env://",
                            rank=global_rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not Path(args.graphs).exists():
        if global_rank == 0:
            print(f"graphs file not found: {args.graphs}\n"
                  f"Build one first:  python -m experimental.graph_dataset --elements U "
                  f"--limit 400 --tag Usubset", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(2)

    main(args, local_rank, global_rank, world_size)
    dist.destroy_process_group()
