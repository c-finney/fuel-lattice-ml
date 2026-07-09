"""
test_composition_parse.py — Tests for parse_composition().

Verifies that UN0.5C0.5, "U1 N0.5 C0.5", and "Ce0.8343 Nd0.1657 O2" all parse
to the expected pymatgen Composition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pymatgen.core import Composition
from engine.reference_resolver import parse_composition


class TestCompositionParse:

    def test_compact_notation(self):
        """UN0.5C0.5 should parse to U:1, N:0.5, C:0.5"""
        comp = parse_composition("UN0.5C0.5")
        assert abs(comp["U"] - 1.0) < 1e-4
        assert abs(comp["N"] - 0.5) < 1e-4
        assert abs(comp["C"] - 0.5) < 1e-4

    def test_spaced_notation(self):
        """'U1 N0.5 C0.5' (spaced) should give same as compact"""
        comp = parse_composition("U1 N0.5 C0.5")
        assert abs(comp["U"] - 1.0) < 1e-4
        assert abs(comp["N"] - 0.5) < 1e-4
        assert abs(comp["C"] - 0.5) < 1e-4

    def test_cendo2_notation(self):
        """'Ce0.8343 Nd0.1657 O2' should parse with correct element fractions"""
        comp = parse_composition("Ce0.8343 Nd0.1657 O2")
        assert abs(comp["Ce"] - 0.8343) < 1e-4
        assert abs(comp["Nd"] - 0.1657) < 1e-4
        assert abs(comp["O"] - 2.0) < 1e-4

    def test_pure_compounds(self):
        """Pure end-members should parse correctly"""
        un = parse_composition("UN")
        assert abs(un["U"] - 1.0) < 1e-4
        assert abs(un["N"] - 1.0) < 1e-4

        uc = parse_composition("UC")
        assert abs(uc["U"] - 1.0) < 1e-4
        assert abs(uc["C"] - 1.0) < 1e-4

    def test_ceo2(self):
        """CeO2 should parse cleanly"""
        comp = parse_composition("CeO2")
        assert abs(comp["Ce"] - 1.0) < 1e-4
        assert abs(comp["O"] - 2.0) < 1e-4

    def test_result_is_composition(self):
        """parse_composition should return a pymatgen Composition"""
        comp = parse_composition("UN0.5C0.5")
        assert isinstance(comp, Composition)
