from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "05_论文图表"
EXPECTED_STEMS = [
    "Fig_1_system_model",
    "Fig_2_workload_validation",
    "Fig_3_binding_gap_heatmap",
    "Fig_4_memory_models",
    "Fig_5_wireless_fairness",
    "Fig_6_convergence_runtime_signaling",
]
FORMATS = {"PNG": ".png", "PDF": ".pdf", "SVG": ".svg", "TIFF": ".tiff"}


def test_each_format_contains_exactly_six_required_figures() -> None:
    for directory, suffix in FORMATS.items():
        actual = sorted(path.name for path in (FIGURE_ROOT / directory).glob(f"*{suffix}"))
        expected = sorted(f"{stem}{suffix}" for stem in EXPECTED_STEMS)
        assert actual == expected


def test_raster_figures_are_nonblank_and_high_resolution() -> None:
    for directory, suffix in [("PNG", ".png"), ("TIFF", ".tiff")]:
        paths = list((FIGURE_ROOT / directory).glob(f"*{suffix}"))
        assert len(paths) == 6
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
                assert image.width >= 1800
                assert image.height >= 1000
                assert float(rgb.std()) > 5.0


def test_vector_figures_are_nonempty() -> None:
    for directory, suffix in [("PDF", ".pdf"), ("SVG", ".svg")]:
        paths = list((FIGURE_ROOT / directory).glob(f"*{suffix}"))
        assert len(paths) == 6
        for path in paths:
            assert path.stat().st_size > 10_000
