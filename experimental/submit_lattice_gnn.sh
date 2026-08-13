#!/bin/bash
# Frontier batch script — multi-node DDP training of the crystal-graph lattice-parameter GNN.
#
# Generated from the frontier-pytorch skill (fuel-agent repo,
# skills/frontier-pytorch/assets/submit_frontier.sh). Both hard-won gotchas are in place:
#   1) NCCL_SOCKET_IFNAME=hsn0 — ONE source of truth; the Python uses setdefault so it never
#      overrides this export. hsn0-only is OLCF's DOCUMENTED REMEDY for multi-node collective
#      hangs; their own default is all four NICs for best bandwidth at scale. We apply the
#      remedy up front (verified against the OLCF doc 2026-08-13) — a deliberate deviation.
#      If this job runs correctly and you want more bandwidth, try hsn0,hsn1,hsn2,hsn3.
#   2) each srun step gets its OWN --master_port — a shared port makes the second step hang
#      on the first step's TCP store lingering in TIME_WAIT. NOTE: this rule is empirical,
#      NOT from the OLCF doc (which recommends 3442 generally and never runs two live steps).
#
# SITE CONFIG: __FRONTIER_ACCOUNT__ and __FRONTIER_ENV__ are PLACEHOLDERS. This file lives in
# a git repo — do NOT commit real values into it. Substitute them in a working copy, or
# submit with `sbatch -A <account> submit_lattice_gnn.sh` (the -A flag overrides the
# directive; the conda activate line must still be substituted).
#SBATCH -A __FRONTIER_ACCOUNT__
#SBATCH -J lattice_gnn
#SBATCH -o logs/%x-%j.o
#SBATCH -e logs/%x-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --network=disable_rdzv_get     # pairs with FI_CXI_RDZV_PROTO=alt_read from the plugin

# ---- Module stack (order matters; verified against the OLCF doc 2026-08-11) ----
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load rccl-net-plugin          # AFTER rocm — enables the Slingshot/OFI high-speed path
module load craype-accel-amd-gfx90a

# Non-default CPE: prepend Cray libs
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# ---- Environment ----
conda activate __FRONTIER_ENV__      # needs torch(ROCm) + torch_geometric + mpi4py

# ---- Rendezvous + RCCL ----
export MASTER_ADDR=$(hostname -i)    # harmless IPv6 c10d errno-97 warnings may appear — ignore
export NCCL_SOCKET_IFNAME=hsn0       # OLCF's documented hang remedy; their default is all
                                     # four (hsn0,hsn1,hsn2,hsn3) for bandwidth at scale.
# export NCCL_DEBUG=INFO             # uncomment to debug; want "Using network OFI"
# export NCCL_DEBUG_SUBSYS=INIT,NET

# ---- MIOpen cache to node-local /tmp (avoids disk I/O / DB lock errors) ----
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs results

# ---- Dataset ----
# Build the graphs on a LOGIN NODE first — compute nodes have no internet, so the Materials
# Project query cannot run here:
#     python -m experimental.graph_dataset --elements U --limit 400 --tag Usubset   # smoke
#     python -m experimental.graph_dataset --tag all                                # full build
GRAPHS=${GRAPHS:-$PWD/Dataset/graphs/graphs_Usubset_free.pt}

if [ ! -f "$GRAPHS" ]; then
    echo "ERROR: graphs file not found: $GRAPHS"
    echo "Build it on a login node first (see comment above)."
    exit 2
fi

# ---- Launch: 2 nodes x 8 GCDs = 16 tasks, 1 GPU/task ----
# -c7 leaves a core per task for the system; --gpu-bind=closest respects NUMA.
# Do NOT use torchrun — it mismaps Frontier's NUMA domains.
srun -N2 -n16 -c7 --gpus-per-task=1 --gpu-bind=closest \
     python3 -W ignore -u experimental/train_lattice_gnn.py \
     --graphs "$GRAPHS" \
     --epochs 200 --batch-size 64 --hidden 128 --num-layers 4 \
     --out-dir results \
     --master_addr=$MASTER_ADDR --master_port=3442

# If you chain a SECOND step (e.g. the raw-distance leakage diagnostic), it needs its OWN
# port — 3443, not 3442. Uncomment to run it:
#
# srun -N2 -n16 -c7 --gpus-per-task=1 --gpu-bind=closest \
#      python3 -W ignore -u experimental/train_lattice_gnn.py \
#      --graphs "${GRAPHS/_free.pt/_raw.pt}" \
#      --epochs 50 --batch-size 64 \
#      --master_addr=$MASTER_ADDR --master_port=3443
