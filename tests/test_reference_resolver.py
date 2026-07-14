"""
test_reference_resolver.py — Tests for endmembers() and resolve_reference().

All MP calls are mocked, with values that MATCH Data/reference_systems.json.
Tests cover:
  - U(N,C) end-members = {UN, UC}
  - UN0.7C0.3 -> N dominant  -> UN
  - UN0.3C0.7 -> C dominant  -> UC
  - UN0.5C0.5 tie -> energy_above_hull is degenerate (both 0.0) -> falls through to
    formation_energy_per_atom -> UN
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
    "n_symmetry_ops": 48,
    "nsites": 2,
    # 0.0, NOT 0.005 — matches Data/reference_systems.json. An earlier mock had
    # 0.005 here, which made the UN/UC tie look decidable on hull energy alone.
    # It is not: UN and UC are each line compounds on their own chemsys hull, so
    # both are at 0.0 and the hull comparison is degenerate. The mock's invented
    # 0.005 meant the tie-break test never exercised the case that actually
    # occurs in the data.
    "energy_above_hull": 0.0,
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
    """Real curated values — UN and UC are BOTH on the hull, so this cannot break their tie."""
    mapping = {
        "UN": 0.0,
        "UC": 0.0,
        "CeO2": 0.0,
        "NdO2": None,
    }
    return mapping.get(formula)


def _mock_formation_energy(formula):
    """Real curated values (eV/atom, lower = more stable) — this is what breaks the UN/UC tie."""
    mapping = {
        "UN": -1.5816823568749996,
        "UC": -0.2554539324999965,
        "CeO2": -3.927176196666666,
        "NdO2": -3.1596579241666647,
    }
    return mapping.get(formula)


class TestResolveReference:

    def test_unc_dominant_endmember(self):
        """UN0.7C0.3 -> N fraction=0.7 dominant -> UN should be chosen"""
        comp = parse_composition("UN0.7C0.3")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull), \
             patch("engine.reference_resolver.mp_client.formation_energy",
                   side_effect=_mock_formation_energy):
            result = resolve_reference(comp)

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-1865"   # UN
        assert "UN" in result["basis"] or "dominant" in result["basis"].lower()

    def test_unc_carbon_dominant_endmember(self):
        """
        UN0.3C0.7 -> C fraction=0.7 dominant -> UC must be chosen.

        This is the half of the rule the UNUC.csv benchmark used to get wrong: every
        row was labelled with UN (mp-1865) regardless of composition, including the
        C-dominant ones.
        """
        comp = parse_composition("UN0.3C0.7")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull), \
             patch("engine.reference_resolver.mp_client.formation_energy",
                   side_effect=_mock_formation_energy):
            result = resolve_reference(comp)

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-2489"   # UC

    def test_unc_tie_broken_by_formation_energy(self):
        """
        UN0.5C0.5 -> stoichiometric tie -> UN wins.

        energy_above_hull CANNOT decide this: UN and UC are both 0.0 (each is a line
        compound on its own chemsys hull). The decision must fall through to
        formation_energy_per_atom, where UN (-1.582) is far more stable than UC (-0.255).
        Before the fallback existed, an equal-hull tie kept 'original order' — i.e. the
        50/50 reference depended on dict iteration order.
        """
        comp = parse_composition("UN0.5C0.5")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull), \
             patch("engine.reference_resolver.mp_client.formation_energy",
                   side_effect=_mock_formation_energy):
            result = resolve_reference(comp)

        assert result["status"] == "ok"
        assert result["mp_id"] == "mp-1865"   # UN — more negative formation energy
        assert "formation_energy" in result["basis"].lower()

    def test_cendo2_existence_guard_fallback(self):
        """
        Ce0.2Nd0.8O2 -> Nd fraction=0.8 dominant -> NdO2 is rejected (non-host)
        -> fallback to CeO2
        """
        comp = parse_composition("Ce0.2 Nd0.8 O2")
        with patch("engine.reference_resolver.mp_client.reference_symmetry",
                   side_effect=_mock_reference_symmetry), \
             patch("engine.reference_resolver.mp_client.energy_above_hull",
                   side_effect=_mock_energy_above_hull), \
             patch("engine.reference_resolver.mp_client.formation_energy",
                   side_effect=_mock_formation_energy):
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
