"""
reference_resolver.py — Dominant end-member + energy_above_hull tie-break + existence guard.

v1 scope: single 2-component mixed sublattice (covers U(N,C), (Ce,Nd)O2).
Multi-sublattice or >2 mixed components -> needs_reference.

Algorithm:
  1. Parse composition -> identify framework vs mixed elements -> end-members
  2. Pick dominant end-member (largest mixed-element fraction)
     Tie (a true 50/50 mix) -> the MORE STABLE end-member wins, decided in order:
       a. lowest energy_above_hull            (mp_client.energy_above_hull)
       b. lowest formation_energy_per_atom    (mp_client.formation_energy)
     (b) is not a formality: end-members from different chemical systems are each
     on their own hull and both read energy_above_hull = 0.0, so (a) cannot
     separate them. UN vs UC is exactly that case.
  3. Existence/compatibility guard: if chosen end-member is not a sane stable host,
     fall back to the other; if neither -> needs_reference
  4. Resolve symmetry via reference_symmetry()

This is the same rule the benchmark CSVs' `ref_mp-id` column follows: the most
prevalent end-member's mp-id, and for a 50/50 mix the more stable one's.
"""

from __future__ import annotations

from pymatgen.core import Composition

from engine import mp_client


# ---------------------------------------------------------------------------
# Composition parsing
# ---------------------------------------------------------------------------

def parse_composition(s: str) -> Composition:
    """
    Parse a composition string tolerating spaces and non-integer stoichiometries.
    Examples: "UN0.5C0.5", "U1 N0.5 C0.5", "Ce0.8343 Nd0.1657 O2"
    """
    return Composition(s)


# ---------------------------------------------------------------------------
# End-member detection
# ---------------------------------------------------------------------------

def endmembers(comp: Composition) -> list[str] | None:
    """
    Identify the single 2-component mixed sublattice and return the two end-members.

    Strategy:
    - Elements with integer (or near-integer within 1e-3) fractional amounts
      in the reduced formula are "framework" elements.
    - Elements with non-integer fractional amounts are "mixed" elements.
    - v1: exactly 2 mixed elements -> two end-members (framework + each mixed element).
    - Otherwise (0, 1, or >2 mixed elements, or complex multi-sublattice) -> None.

    Returns list of two formula strings, or None.
    """
    # Work with reduced composition
    red = comp.reduced_composition
    amounts = dict(red.as_dict())  # element_symbol -> amount (floats)

    # Identify mixed elements (non-integer amounts)
    framework = {}
    mixed = {}
    for el, amt in amounts.items():
        if abs(amt - round(amt)) < 1e-3:
            framework[el] = round(amt)
        else:
            mixed[el] = amt

    if len(mixed) != 2:
        return None  # v1 only handles exactly 2 mixed elements

    mixed_els = list(mixed.keys())
    mixed_amts = list(mixed.values())

    # Total mixed amount (should sum to ~ integer)
    total_mixed = sum(mixed_amts)
    int_total = round(total_mixed)
    if abs(total_mixed - int_total) > 0.05:
        return None  # Cannot renormalize cleanly

    # Build end-members — use pymatgen Composition for canonical formula ordering
    end_members = []
    for el, amt in zip(mixed_els, mixed_amts):
        parts = {}
        parts.update(framework)
        parts[el] = int_total  # replace mixed with integer count
        # Build canonical formula via pymatgen (gives e.g. CeO2 not O2Ce)
        em_comp = Composition(parts)
        formula = em_comp.reduced_formula
        end_members.append(formula)

    return end_members


# ---------------------------------------------------------------------------
# Existence guard
# ---------------------------------------------------------------------------

# Known non-hosts: stable phase is different from the nominal end-member
_NON_HOSTS: dict[str, str] = {
    # NdO2 does not exist as a stable fluorite (Nd is 3+); real stable phase is Nd2O3
    "NdO2": "NdO2 is not a stable fluorite host (Nd is 3+; stable phase is hexagonal Nd2O3)",
}


def _is_valid_host(formula: str) -> tuple[bool, str]:
    """
    Return (is_valid, reason).
    Checks the known non-host list; future: could add MP stability check.
    """
    # Normalise formula to remove coefficients of 1
    try:
        comp = Composition(formula)
        norm_formula = comp.reduced_formula
    except Exception:
        norm_formula = formula

    for non_host, reason in _NON_HOSTS.items():
        try:
            nh_formula = Composition(non_host).reduced_formula
        except Exception:
            nh_formula = non_host
        if norm_formula == nh_formula:
            return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_reference(
    comp: Composition,
    explicit_ref: str | None = None,
) -> dict:
    """
    Resolve the reference end-member for *comp*.

    Parameters
    ----------
    comp         : pymatgen Composition of the solid solution
    explicit_ref : if provided (mp-id or formula), use it directly

    Returns
    -------
    {"status": "ok", "mp_id": ..., "crystal_system": ..., ...symmetry...,
     "source": ..., "basis": ...}
    OR
    {"status": "needs_reference", "candidates": [...], "reason": ...}
    """
    # ----- Explicit reference (user-supplied or CSV ref_mp-id) -----
    if explicit_ref:
        sym = mp_client.reference_symmetry(explicit_ref)
        if sym is None:
            return {
                "status": "needs_reference",
                "candidates": [],
                "reason": f"Could not resolve explicit reference '{explicit_ref}' from MP.",
            }
        return {
            "status": "ok",
            "basis": f"explicit reference: {explicit_ref}",
            "source": sym.get("_source", "unknown"),
            **{k: v for k, v in sym.items() if not k.startswith("_")},
        }

    # ----- Auto-resolution -----
    ems = endmembers(comp)
    if ems is None:
        return {
            "status": "needs_reference",
            "candidates": [],
            "reason": (
                "Cannot auto-resolve reference: composition has 0, 1, or >2 mixed "
                "elements (v1 supports exactly 2-component mixed sublattice). "
                "Provide --reference <mp-id|formula>."
            ),
        }

    # Identify the mixed elements and their fractions in the ORIGINAL comp
    red = comp.reduced_composition
    amounts = dict(red.as_dict())
    framework = {el: amt for el, amt in amounts.items() if abs(amt - round(amt)) < 1e-3}
    mixed = {el: amt for el, amt in amounts.items() if abs(amt - round(amt)) >= 1e-3}

    mixed_els = list(mixed.keys())
    mixed_amts = list(mixed.values())

    # Map end-member formula -> its distinguishing mixed element's fraction
    em_fractions: dict[str, float] = {}
    for em_formula in ems:
        try:
            em_comp = Composition(em_formula)
            em_els = set(el.symbol for el in em_comp.elements)
            # The distinguishing element is the mixed element present in this end-member
            for el, amt in zip(mixed_els, mixed_amts):
                if el in em_els:
                    em_fractions[em_formula] = amt
                    break
        except Exception:
            em_fractions[em_formula] = 0.0

    # Sort by fraction descending -> dominant first
    sorted_ems = sorted(em_fractions, key=lambda f: em_fractions[f], reverse=True)

    # Tie-break by energy_above_hull — lower wins
    if len(sorted_ems) >= 2:
        f1, f2 = sorted_ems[0], sorted_ems[1]
        frac1, frac2 = em_fractions[f1], em_fractions[f2]
        if abs(frac1 - frac2) < 1e-4:
            # Stoichiometric tie (a true 50/50 mix) -> the more STABLE end-member wins.
            #
            # energy_above_hull is tried first, but it frequently CANNOT decide:
            # two end-members from different chemical systems are each on their own
            # hull, so both read 0.0. That is exactly the U(N,C) case (UN and UC).
            # Falling through to "keep original order" there would make the 50/50
            # reference depend on dict iteration order — silently arbitrary.
            # formation_energy_per_atom is the real discriminator: UN -1.582 vs
            # UC -0.255 eV/atom.
            e1 = mp_client.energy_above_hull(f1)
            e2 = mp_client.energy_above_hull(f2)

            hull_decided = (
                e1 is not None and e2 is not None and abs(e1 - e2) > 1e-6
            )
            if hull_decided:
                sorted_ems = [f1, f2] if e1 < e2 else [f2, f1]
                tie_basis = (
                    f"stoichiometric tie broken by energy_above_hull "
                    f"({sorted_ems[0]}: {min(e1, e2):.4f} eV/atom)"
                )
            else:
                # Hull energies are equal (or unknown) -> fall back to formation energy.
                g1 = mp_client.formation_energy(f1)
                g2 = mp_client.formation_energy(f2)
                if g1 is not None and g2 is not None:
                    sorted_ems = [f1, f2] if g1 <= g2 else [f2, f1]
                    tie_basis = (
                        f"stoichiometric tie; energy_above_hull equal "
                        f"({e1} vs {e2}) -> broken by formation_energy_per_atom "
                        f"({sorted_ems[0]}: {min(g1, g2):.4f} eV/atom, more stable)"
                    )
                elif g1 is not None:
                    tie_basis = f"stoichiometric tie; only {f1} formation energy known"
                elif g2 is not None:
                    sorted_ems = [f2, f1]
                    tie_basis = f"stoichiometric tie; only {f2} formation energy known"
                else:
                    tie_basis = (
                        f"stoichiometric tie; no stability data — defaulting to "
                        f"first: {sorted_ems[0]}"
                    )
        else:
            tie_basis = f"dominant mixed element ({sorted_ems[0]}, fraction={frac1:.4f})"
    else:
        tie_basis = f"single end-member: {sorted_ems[0]}"

    # Attempt host resolution in preference order
    errors = []
    for candidate in sorted_ems:
        valid, reason = _is_valid_host(candidate)
        if not valid:
            errors.append(f"{candidate}: {reason}")
            continue  # try the other end-member

        sym = mp_client.reference_symmetry(candidate)
        if sym is None:
            errors.append(f"{candidate}: not found in MP or curated table")
            continue

        return {
            "status": "ok",
            "basis": tie_basis,
            "source": sym.get("_source", "unknown"),
            **{k: v for k, v in sym.items() if not k.startswith("_")},
            "_chosen_endmember": candidate,
        }

    # Both end-members failed
    return {
        "status": "needs_reference",
        "candidates": ems,
        "reason": (
            f"Could not resolve a valid reference for end-members {ems}. "
            f"Issues: {'; '.join(errors)}. "
            "Provide --reference <mp-id|formula>."
        ),
    }
