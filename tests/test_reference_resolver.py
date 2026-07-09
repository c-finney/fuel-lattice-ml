"""
test_reference_resolver.py — Tests for endmembers() and resolve_reference().

All MP calls are mocked. Tests cover:
  - U(N,C) end-members = {UN, UC}
  - UN0.5C0.5 tie -> D13 picks by energy_above_hull (lower wins)
  - Ce0.2Nd0.8O2 existence guard -> NdO2 rejected -> falls back to CeO2
  - size-3 mixed set -> needs_reference
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch, MagicMock
import pytest

from engine.reference_resolver import (
    parse_composition,
    endmembers,
    resolve_reference,
)


# ---------------------------------------------------------------------------
# endmembers() tests
# ---------------------------------------------------------------------------

class TestEndmembers:

    def test_unc_endmembers(self):
        """UN0.5C0.5 -> end-members should be UN and UC"""
        comp = parse_composition("UN0.5C0.5")
        ems = endmembers(comp)
        assert ems is not None
        assert len(ems) == 2
        # Both should contain U, and one N, one C
        assert any("N" in e and "C" not in e for e in ems), f"No UN-like end-member in {ems}"
        assert any("C" in e and "N" not in e for e in ems), f"No UC-like end-member in {ems}"

    def test_cendo2_endmembers(self):
        """Ce0.8Nd0.2O2 -> end-members should be CeO2 and NdO2"""
        comp = parse_composition("Ce0.8 Nd0.2 O2")
        ems = endmembers(comp)
        assert ems is not None
        assert len(ems) == 2
        assert any("Ce" in e and "Nd" not in e for e in ems)
        assert any("Nd" in e and "Ce" not in e for e in ems)

    def test_pure_compound_returns_none(self):
        """UN (pure) -> no mixed elements -> None"""
        comp = parse_composition("UN")
        result = endmembers(comp)
        assert result is None

    def test_three_mixed_elements_returns_none(self):
        """U(N,C,O) -> 3 mixed elements -> None (v1 scope)"""
        comp = parse_composition("U1 N0.33 C0.33 O0.34")
        result = endmembers(comp)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_reference() tests — MP mocked
# ---------------------------------------------------------------------------

_UN_SYMMETRY = {
    "mp_id": "mp-1865",
    "crystal_system": "cubic",
    "spacegroup_num": 225,
    "is_centrosymmetric": True,
    "n_symmetry_ops": 48,
    "nsites": 2,
    "energy_above_hull": 0.0,
    "_source": "curated_table",
}

_UC_SYMMETRY = {
    "mp_id": "mp-2489",
    "crystal_system": "cubic",
    "spacegroup_num": 225,
    "is_centrosymmetric": True,
    "n_symmetry_ops": 96,
    "nsites": 2,
    "energy_above_hull": 0.005,
    "_source": "curated_table",
}

_CEO2_SYMMETRY = {
    "mp_id": "mp-20194",
    "crystal_system": "cubic",
    "spacegroup_num": 225,
    "is_centrosymmetric": True,
    "n_symmetry_ops": 48,
    "nsites": 12,
    "energy_above_hull": 0.0,
    "_source": "curated_table",
}


def _mock_reference_symmetry(formula_or_mpid):
    """Route mock by formula."""
    mapping = {
        "UN": _UN_SYMMETRY,
        "mp-1865": _UN_SYMMETRY,
        "UC": _UC_SYMMETRY,
        "mp-2489": _UC_SYMMETRY,
        "CeO2": _CEO2_SYMMETRY,
        "mp-20194": _CEO2_SYMMETRY,
        "NdO2": None,   # not a valid host
    }
    return mapping.get(formula_or_mpid)


def _mock_energy_above_hull(formula):
    mapping = {
        "UN": 0.0,
        "UC": 0.005,
        "CeO2": 0.0,
        "NdO2": None,
    }
    return mapping.get(formula)


class TestResolveReference:

    def test_unc_dominant_endmember(self):
        """UN0.7C0.3 -> N fraction=0.7 dominant -> UN should be chosen"""
        comp = parse_composition("UN0.7C0.3")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull):
            result = resolve_reference(comp)

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-1865"   # UN
        assert "UN" in result["basis"] or "dominant" in result["basis"].lower()

    def test_unc_tie_broken_by_energy(self):
        """UN0.5C0.5 -> stoichiometric tie -> picks UN (lower energy_above_hull=0.0 vs UC=0.005)"""
        comp = parse_composition("UN0.5C0.5")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull):
            result = resolve_reference(comp)

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-1865"   # UN wins (energy=0.0 < UC energy=0.005)
        assert "energy_above_hull" in result["basis"].lower() or "tie" in result["basis"].lower()

    def test_cendo2_existence_guard_fallback(self):
        """
        Ce0.2Nd0.8O2 -> Nd fraction=0.8 dominant -> NdO2 is rejected (non-host)
        -> fallback to CeO2
        """
        comp = parse_composition("Ce0.2 Nd0.8 O2")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull):
            result = resolve_reference(comp)

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-20194"  # CeO2 fallback

    def test_three_mixed_elements_needs_reference(self):
        """U(N,C,O) -> >2 mixed -> needs_reference"""
        comp = parse_composition("U1 N0.33 C0.33 O0.34")
        result = resolve_reference(comp)
        assert result["status"] == "needs_reference"

    def test_explicit_reference_bypasses_resolution(self):
        """Explicit reference mp-id should be used directly"""
        comp = parse_composition("UN0.5C0.5")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry):
            result = resolve_reference(comp, explicit_ref="mp-2489")

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-2489"   # UC (forced)

    def test_pure_compound_needs_reference(self):
        """Pure compound (no mixed elements) -> needs_reference"""
        comp = parse_composition("UN")
        result = resolve_reference(comp)
        assert result["status"] == "needs_reference"
