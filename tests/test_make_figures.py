"""
test_make_figures.py — The manuscript figures can be redrawn from the deposited
result files alone, with no model binary and no rebuilt dataset, so this runs in
a clean clone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import make_figures


def test_figures_2_3_4_are_drawn_from_the_deposited_files(tmp_path):
    written = make_figures.main(["--out-dir", str(tmp_path)])
    assert [p.name for p in written] == [
        "figure2_cross_validation_rf1.png",
        "figure3_validation_systems.png",
        "figure4_spearman_cubic.png",
    ]
    for p in written:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 50_000
    assert sorted(x.name for x in tmp_path.iterdir()) == sorted(p.name for p in written)
