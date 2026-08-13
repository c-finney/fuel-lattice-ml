"""
graph_dataset.py — crystal-graph dataset builder for the lattice-parameter GNN.

WHY THIS EXISTS
---------------
The committed seed (`Data/MP_Dataset_Original_Trimmed.csv`) keeps only scalars: the
Materials Project `structure` is fed to SpacegroupAnalyzer and then dropped (see
config.QUERY_FIELDS). A crystal-graph network needs the geometry, so this module re-queries
MP and persists graphs instead of throwing the structures away.

PARITY WITH THE RF/GBR BASELINE (deliberate, do not drift)
----------------------------------------------------------
Same target and same filters as engine/train_models.prepare_training_frame():
  - target  Y = [a, b, c] from the CONVENTIONAL standard cell
  - filter  a, b, c <= config.TRAIN_THRESH (10 A)
  - dedup   single-element polymorphs only, keeping lowest formation_energy_per_atom,
            on (composition_reduced, spacegroup_num, nsites)
  - noble gases excluded (build_dataset.NOBLE_GASES)
so a GNN MAE_cubic is directly comparable to rf1 (0.121701 A) and gbr1 (0.113556 A).

>>> TARGET LEAKAGE — READ THIS BEFORE TRUSTING ANY NUMBER <<<
A crystal graph built from the relaxed structure encodes the answer: the interatomic
distances ARE the lattice, so a network given raw distances can read a, b, c straight off
its edge features and will post an absurdly low MAE that means nothing. The RF baseline
never sees geometry (composition + crystal system + spacegroup only), so a raw-distance GNN
is not a fair comparison — it is a different, trivial problem.

Two edge modes exist for exactly this reason:

  edge_scale="free"  (DEFAULT, the real task)
      Distances are divided by the structure's own mean nearest-neighbour distance, then
      Gaussian-expanded. Edge features carry SHAPE and TOPOLOGY but no absolute scale. The
      model must infer the absolute lattice constant from species identity + local
      coordination, which is the physically meaningful problem and IS comparable to RF.

  edge_scale="raw"   (DIAGNOSTIC ONLY — leaks the target)
      Raw angstrom distances, which the network can in principle integrate back into the
      cell dimensions. Never report a "raw" number as a model result, and never compare it
      to the RF baseline.

      MEASURED 2026-08-11 (188 training graphs, 8 epochs, CPU): raw reached val MAE
      1.9671 A vs free's 1.6678 A — i.e. the two modes did NOT separate, and raw was not
      better. That budget is far too small to converge either mode, so this is NOT evidence
      that raw is leak-free; it only means the smoke run cannot tell them apart. Re-run the
      comparison at a real training budget before drawing any conclusion about how much the
      raw mode leaks in practice.

PREDICTION-TIME CAVEAT
----------------------
Unlike the RF pipeline, this model needs a structure at inference. A novel solid solution
(e.g. UN0.5C0.5) has no MP entry, so you must supply a TEMPLATE: take the reference host
structure (Data/reference_systems.json / mp_client.reference_symmetry) and substitute the
species. With edge_scale="free" the template's absolute scale is normalised away, which is
what makes template substitution legitimate — but it is still an assumption about the
prototype, and it should be stated in any result.

STATUS: written 2026-08-11. Smoke-tested on a capped U-containing subset. NOT yet run at
full scale, and NOT yet trained on Frontier.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from engine import config

# Kept in sync with build_dataset.NOBLE_GASES (imported lazily to avoid a hard cycle).
NOBLE_GASES = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"})

# Graph construction defaults — CGCNN conventions.
CUTOFF_A = 8.0          # neighbour search radius, angstrom
MAX_NEIGHBORS = 12      # keep the 12 nearest within the cutoff
GAUSS_CENTERS = 40      # Gaussian basis for the distance expansion
MAX_Z = 100             # embedding table size (Z=1..99 covers everything post noble-gas cut)

GRAPH_DIR = config.DATASETS / "graphs"


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------
# Continuous per-element descriptors. Deliberately small and scale-relevant: the model
# predicts a LENGTH, so ionic/atomic radius and electronegativity are the physics that
# matters. Element identity itself goes in as an embedding index (z), not a one-hot.
_ELEM_FEATURE_NAMES = ["X", "atomic_radius", "atomic_mass", "row", "group", "valence_e"]


def _element_features(el) -> list[float]:
    """Six scalars per element, with explicit fallbacks for missing pymatgen data."""
    def _f(v, default=0.0):
        try:
            if v is None:
                return default
            v = float(v)
            return default if math.isnan(v) else v
        except (TypeError, ValueError):
            return default

    radius = el.atomic_radius if el.atomic_radius is not None else el.atomic_radius_calculated
    try:
        valence = float(sum(n for _, _, n in el.full_electronic_structure[-2:]))
    except Exception:
        valence = 0.0

    return [
        _f(el.X, 1.5),              # Pauling electronegativity
        _f(radius, 1.4),            # angstrom
        _f(el.atomic_mass),
        _f(el.row),
        _f(el.group),
        valence,
    ]


def _gaussian_expand(distances, centers, width):
    import torch
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))


# ---------------------------------------------------------------------------
# Structure -> PyG Data
# ---------------------------------------------------------------------------
def structure_to_graph(
    structure,
    y=None,
    crystal_system: str | None = None,
    spacegroup_num: int | None = None,
    edge_scale: str = "free",
    cutoff: float = CUTOFF_A,
    max_neighbors: int = MAX_NEIGHBORS,
):
    """Convert a pymatgen Structure into a torch_geometric Data object.

    Returns None if the structure yields no edges within `cutoff` (isolated atoms), which
    would otherwise produce a graph the convolutions cannot use.
    """
    import torch
    from torch_geometric.data import Data

    if edge_scale not in ("free", "raw"):
        raise ValueError(f"edge_scale must be 'free' or 'raw', got {edge_scale!r}")

    n = len(structure)
    z = torch.tensor([min(site.specie.Z, MAX_Z - 1) for site in structure], dtype=torch.long)
    x_cont = torch.tensor(
        [_element_features(site.specie.element if hasattr(site.specie, "element") else site.specie)
         for site in structure],
        dtype=torch.float,
    )

    # Neighbour list — periodic images included by get_all_neighbors.
    all_nbrs = structure.get_all_neighbors(cutoff, include_index=True)
    src, dst, dists = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        nbrs = sorted(nbrs, key=lambda nb: nb.nn_distance)[:max_neighbors]
        for nb in nbrs:
            src.append(i)
            dst.append(nb.index)
            dists.append(nb.nn_distance)

    if not dists:
        return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    d = torch.tensor(dists, dtype=torch.float)

    # --- the leakage control (see module docstring) ---
    if edge_scale == "free":
        # Divide by this structure's own mean nearest-neighbour distance. The result is
        # dimensionless: it describes coordination geometry, not size.
        per_site_min = {}
        for i, dd in zip(src, dists):
            per_site_min[i] = min(per_site_min.get(i, float("inf")), dd)
        scale = sum(per_site_min.values()) / max(len(per_site_min), 1)
        scale = scale if scale > 1e-6 else 1.0
        d_feat = d / scale
        span = 3.0          # d/d_min typically lands in [1, 3]
    else:
        d_feat = d
        span = cutoff

    centers = torch.linspace(0.0, span, GAUSS_CENTERS)
    width = float(span / GAUSS_CENTERS)
    edge_attr = _gaussian_expand(d_feat, centers, width)

    # Graph-level descriptors the RF also gets: crystal system one-hot + spacegroup + nsites.
    cs_vec = [0.0] * len(config.CS_LABELS)
    if crystal_system:
        key = f"cs_{str(crystal_system).lower()}"
        if key in config.CS_LABELS:
            cs_vec[config.CS_LABELS.index(key)] = 1.0
    u = torch.tensor(
        [cs_vec + [float(spacegroup_num or 0) / 230.0, float(n) / 100.0]],
        dtype=torch.float,
    )

    data = Data(x=x_cont, z=z, edge_index=edge_index, edge_attr=edge_attr, u=u)
    if y is not None:
        data.y = torch.tensor([list(y)], dtype=torch.float)   # [1, 3] -> a, b, c
    return data


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_graphs(
    limit: int | None = None,
    elements: list[str] | None = None,
    edge_scale: str = "free",
    thresh: float = config.TRAIN_THRESH,
    out_dir: Path | None = None,
    tag: str = "all",
    progress=print,
) -> dict:
    """Query MP, build crystal graphs, save them under Dataset/graphs/ (gitignored).

    Parameters
    ----------
    limit     : cap on MP docs — smoke-test knob. None = the whole database (hours, and
                tens of GB of structures over the wire).
    elements  : restrict the MP query to materials containing ALL of these elements,
                e.g. ["U"]. This is the cheap way to get a meaningful subset.
    edge_scale: "free" (default, honest) or "raw" (leakage diagnostic). See module docstring.
    thresh    : lattice filter in angstrom, matched to config.TRAIN_THRESH for RF parity.
    tag       : filename tag, so a U-only smoke build doesn't overwrite a full build.
    """
    import torch
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from engine import mp_client

    out_dir = Path(out_dir) if out_dir else GRAPH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filters = {"deprecated": False}
    if elements:
        filters["elements"] = list(elements)

    progress(f"[graphs] querying MP (elements={elements or 'ALL'})…")
    docs = mp_client.search_summary(fields=config.QUERY_FIELDS, **filters)
    progress(f"[graphs]   {len(docs)} docs returned")

    docs = [d for d in docs if not (NOBLE_GASES & {el.symbol for el in d.elements})]
    progress(f"[graphs]   {len(docs)} after noble-gas filter")

    if limit:
        docs = docs[:limit]
        progress(f"[graphs]   capped to {len(docs)} (smoke test)")

    # --- pass 1: metadata only, so dedup/filter happen BEFORE expensive graph building ---
    rows = []
    for i, doc in enumerate(docs):
        if i % 250 == 0:
            progress(f"[graphs]   symmetry {i}/{len(docs)}…")
        try:
            analyzer = SpacegroupAnalyzer(doc.structure, symprec=0.01)
            conventional = analyzer.get_conventional_standard_structure()
            lat = conventional.lattice
            rows.append({
                "idx": i,
                "material_id": str(doc.material_id),
                "composition_reduced": str(doc.composition_reduced),
                "nelements": doc.nelements,
                "nsites": doc.nsites,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "a": lat.a, "b": lat.b, "c": lat.c,
                "crystal_system": analyzer.get_crystal_system(),
                "spacegroup_num": analyzer.get_space_group_number(),
                "_conventional": conventional,
            })
        except Exception as exc:      # noqa: BLE001 — one bad doc must not kill the build
            progress(f"[graphs]   skip {doc.material_id}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no usable structures returned from MP")

    n0 = len(df)

    # Dedup — identical to prepare_training_frame(): single-element polymorphs only.
    df = df.sort_values("formation_energy_per_atom", ascending=True)
    dup_mask = df.duplicated(subset=["composition_reduced", "spacegroup_num", "nsites"],
                             keep="first")
    df = df.loc[~((df["nelements"] == 1) & dup_mask)].reset_index(drop=True)
    n1 = len(df)

    # Lattice filter — identical threshold to the RF training frame.
    df = df[(df[["a", "b", "c"]] <= thresh).all(axis=1)].reset_index(drop=True)
    n2 = len(df)
    progress(f"[graphs]   {n0} -> {n1} after dedup -> {n2} after <= {thresh} A filter")

    # --- pass 2: graphs ---
    graphs, skipped = [], 0
    for i, row in df.iterrows():
        if i % 250 == 0:
            progress(f"[graphs]   graph {i}/{len(df)}…")
        g = structure_to_graph(
            row["_conventional"],
            y=(row["a"], row["b"], row["c"]),
            crystal_system=row["crystal_system"],
            spacegroup_num=row["spacegroup_num"],
            edge_scale=edge_scale,
        )
        if g is None:
            skipped += 1
            continue
        g.material_id = row["material_id"]
        g.crystal_system = row["crystal_system"]
        graphs.append(g)

    path = out_dir / f"graphs_{tag}_{edge_scale}.pt"
    torch.save(graphs, path)

    meta = {
        "tag": tag,
        "edge_scale": edge_scale,
        "elements_filter": elements,
        "limit": limit,
        "thresh_angstrom": thresh,
        "docs_queried": len(docs),
        "after_dedup": n1,
        "after_lattice_filter": n2,
        "graphs_written": len(graphs),
        "skipped_no_edges": skipped,
        "cutoff_angstrom": CUTOFF_A,
        "max_neighbors": MAX_NEIGHBORS,
        "node_cont_features": _ELEM_FEATURE_NAMES,
        "edge_dim": GAUSS_CENTERS,
        "u_dim": len(config.CS_LABELS) + 2,
        "target": ["a", "b", "c"],
        "parity_note": "filters match engine/train_models.prepare_training_frame()",
    }
    (out_dir / f"graphs_{tag}_{edge_scale}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    progress(f"[graphs] wrote {len(graphs)} graphs -> {path}")
    return {"path": str(path), **meta}


def load_graphs(path):
    import torch
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build crystal graphs for the lattice GNN")
    p.add_argument("--limit", type=int, default=None, help="cap MP docs (smoke test)")
    p.add_argument("--elements", nargs="*", default=None,
                   help="restrict to materials containing ALL of these, e.g. --elements U")
    p.add_argument("--edge-scale", choices=["free", "raw"], default="free",
                   help="'free' = scale-normalised (honest). 'raw' = LEAKS the target.")
    p.add_argument("--tag", type=str, default="all", help="filename tag")
    p.add_argument("--out-dir", type=str, default=None)
    a = p.parse_args()

    if a.edge_scale == "raw":
        print("!! edge_scale=raw LEAKS the target — diagnostic only, never a reportable result.")

    build_graphs(limit=a.limit, elements=a.elements, edge_scale=a.edge_scale,
                 tag=a.tag, out_dir=a.out_dir)
