"""
mp_client.py — Materials Project API wrapper.

Provides:
  - mpr()                    context manager yielding MPRester
  - search_summary()         thin wrapper around mpr().materials.summary.search
  - extract_symmetry()       SpacegroupAnalyzer → symmetry dict
  - reference_symmetry()     curated table → runtime cache → live MP
  - energy_above_hull()      MP lookup, used by reference_resolver.py to
                              break a stoichiometric tie between two
                              end-members (lower energy_above_hull wins)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from engine import config


# ---------------------------------------------------------------------------
# MPRester context manager
# ---------------------------------------------------------------------------

@contextmanager
def mpr():
    """Yield an MPRester instance loaded from .env MP_API_KEY."""
    from mp_api.client import MPRester as _MPRester
    key = config.load_mp_key()
    with _MPRester(key) as client:
        yield client


def search_summary(fields: list[str], **filters: Any) -> list:
    """Call mpr().materials.summary.search with given fields and filters."""
    with mpr() as client:
        return client.materials.summary.search(fields=fields, **filters)


# ---------------------------------------------------------------------------
# Symmetry extraction (shared by build and predict paths)
# ---------------------------------------------------------------------------

def extract_symmetry(structure, nsites: int) -> dict:
    """
    Run SpacegroupAnalyzer on *structure* (conventional cell) and return
    a dict of symmetry + lattice features.

    nsites is passed in as the raw doc.nsites from Materials Project — NOT
    the conventional cell's site count that SpacegroupAnalyzer would give.
    This is an inconsistency, but it is the one the shipped model was
    trained on; changing it would invalidate that model.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
    conventional = analyzer.get_conventional_standard_structure()
    lat = conventional.lattice

    return {
        "a":                   lat.a,
        "b":                   lat.b,
        "c":                   lat.c,
        "alpha":               lat.alpha,
        "beta":                lat.beta,
        "gamma":               lat.gamma,
        "crystal_system":      analyzer.get_crystal_system(),
        "spacegroup_num":      analyzer.get_space_group_number(),
        "is_centrosymmetric":  analyzer.is_laue(),
        "n_symmetry_ops":      len(analyzer.get_space_group_operations()),
        "nsites":              nsites,   # raw doc.nsites — see note above
    }


# ---------------------------------------------------------------------------
# Reference structure cache + resolution
# ---------------------------------------------------------------------------

def _cache_path(mp_id: str) -> Path:
    return config.CACHE_DIR / f"{mp_id}.json"


def _load_cache(mp_id: str) -> dict | None:
    p = _cache_path(mp_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(mp_id: str, data: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(mp_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _curated_lookup(formula_or_mpid: str) -> dict | None:
    """
    Check Data/reference_systems.json for a matching entry.
    Accepts either an mp-id or an end-member formula (case-insensitive).

    This table currently has 9 curated hosts (UN, UC, CeO2, UO2, PuO2, ThO2,
    ZrO2, Nd2O3, NdO2) with full symmetry fields, which is why prediction for
    any of them — by formula, by mp-id, or via automatic end-member
    resolution — works with zero Materials Project API calls.
    """
    if not config.REFERENCE_SYSTEMS.exists():
        return None
    try:
        db = json.loads(config.REFERENCE_SYSTEMS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    key = formula_or_mpid.strip()
    # Direct key lookup (by formula)
    if key in db:
        entry = dict(db[key])
        entry["_source"] = "curated_table"
        return entry
    # Lookup by mp-id
    for formula, entry in db.items():
        if entry.get("mp_id") == key:
            result = dict(entry)
            result["_source"] = "curated_table"
            result["_formula"] = formula
            return result
    return None


def reference_symmetry(formula_or_mpid: str) -> dict | None:
    """
    Resolve symmetry for a reference end-member or mp-id.

    Resolution order:
      1. Data/reference_systems.json (curated — offline, no MP key needed)
      2. Dataset/cache/ref_structures/<mp-id>.json (runtime cache, gitignored)
      3. Live MP query (then cached)

    Returns a dict with keys:
      mp_id, crystal_system, spacegroup_num, is_centrosymmetric,
      n_symmetry_ops, nsites, energy_above_hull
    Or None on failure.
    """
    # 1. Curated table
    curated = _curated_lookup(formula_or_mpid)
    if curated is not None:
        mp_id = curated.get("mp_id")
        # Fill in symmetry from cache/live if any curated field is missing
        needed = {"crystal_system", "spacegroup_num", "is_centrosymmetric",
                  "n_symmetry_ops", "nsites"}
        if needed.issubset(curated.keys()) and all(curated.get(k) is not None for k in needed):
            return curated
        # Fall through to enrich from cache/live using the mp_id
        if mp_id:
            live = _resolve_by_mpid(mp_id)
            if live:
                curated.update({k: v for k, v in live.items() if curated.get(k) is None})
            return curated

    # 2. Determine if it looks like an mp-id
    key = formula_or_mpid.strip()
    if key.startswith("mp-") or key.startswith("mvc-"):
        cached = _load_cache(key)
        if cached is not None:
            return cached
        return _resolve_by_mpid(key)

    # 3. Treat as formula: search MP
    return _resolve_by_formula(key)


def _resolve_by_mpid(mp_id: str) -> dict | None:
    """Fetch symmetry for a known mp-id, using cache first."""
    cached = _load_cache(mp_id)
    if cached is not None:
        return cached

    try:
        docs = search_summary(
            fields=["material_id", "structure", "nsites", "energy_above_hull"],
            material_ids=[mp_id],
        )
    except Exception:
        return None

    if not docs:
        return None

    doc = docs[0]
    sym = extract_symmetry(doc.structure, doc.nsites)
    result = {
        "mp_id":               mp_id,
        "energy_above_hull":   getattr(doc, "energy_above_hull", None),
        "_source":             "live_mp",
        **sym,
    }
    _save_cache(mp_id, result)
    return result


def _resolve_by_formula(formula: str) -> dict | None:
    """
    Search MP by formula, pick the entry with lowest energy_above_hull
    (prefer non-theoretical), cache, and return symmetry.
    """
    try:
        from pymatgen.core import Composition
        comp = Composition(formula)
        # Use chemsys search for reliability
        chemsys = "-".join(sorted(el.symbol for el in comp.elements))
        docs = search_summary(
            fields=["material_id", "structure", "nsites",
                    "energy_above_hull", "theoretical", "composition_reduced"],
            chemsys=chemsys,
        )
    except Exception:
        return None

    if not docs:
        return None

    # Filter to matching reduced composition
    try:
        reduced = str(comp.reduced_formula)
    except Exception:
        reduced = formula

    candidates = [
        d for d in docs
        if str(getattr(d, "composition_reduced", "")).replace(" ", "") == reduced.replace(" ", "")
    ]
    if not candidates:
        candidates = docs  # fallback: all in chemsys

    # Sort: lowest energy_above_hull wins (experimental/theoretical is irrelevant —
    # the theoretical flag causes wrong polymorphs to be preferred, e.g. hexagonal UO2
    # over fluorite UO2, when the experimental structure happens to have higher energy).
    def sort_key(d):
        e = getattr(d, "energy_above_hull", 1e9)
        return float(e) if e is not None else 1e9

    candidates.sort(key=sort_key)
    doc = candidates[0]

    sym = extract_symmetry(doc.structure, doc.nsites)
    mp_id = doc.material_id
    result = {
        "mp_id":             mp_id,
        "energy_above_hull": getattr(doc, "energy_above_hull", None),
        "_source":           "live_mp",
        **sym,
    }
    _save_cache(mp_id, result)
    return result


def energy_above_hull(formula: str) -> float | None:
    """
    Return the energy_above_hull for the most stable MP entry matching
    *formula*. Used by reference_resolver.resolve_reference() to break a
    stoichiometric tie between two end-members — the one with the LOWER
    energy_above_hull wins.
    """
    sym = _curated_lookup(formula)
    if sym is not None and sym.get("energy_above_hull") is not None:
        return float(sym["energy_above_hull"])

    # Live lookup
    result = _resolve_by_formula(formula)
    if result and result.get("energy_above_hull") is not None:
        return float(result["energy_above_hull"])
    return None
