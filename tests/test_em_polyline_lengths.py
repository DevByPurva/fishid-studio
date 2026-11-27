import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoramento_em import compute_polyline_lengths, measure_mask_length  # noqa: E402


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([], (0.0, 0.0)),
        ([np.array([5.0, 5.0])], (0.0, 0.0)),
        ([np.array([0.0, 0.0]), np.array([3.0, 4.0])], (5.0, 5.0)),
        (
            [np.array([0.0, 0.0]), np.array([0.0, 3.0]), np.array([4.0, 3.0])],
            (7.0, 5.0),
        ),
        (
            [
                np.array([0.0, 0.0]),
                np.array([2.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 2.0]),
            ],
            (6.0, 2.0),
        ),
    ],
    ids=["empty", "single-point", "straight", "right-angle", "folded-back"],
)
def test_compute_polyline_lengths(points, expected):
    assert compute_polyline_lengths(points) == pytest.approx(expected)


def test_compute_polyline_lengths_never_shorter_than_direct():
    rng = np.random.default_rng(7)
    for _ in range(20):
        points = rng.uniform(0.0, 200.0, size=(rng.integers(2, 12), 2))
        curve_len, direct_len = compute_polyline_lengths(points)
        assert curve_len >= direct_len >= 0.0


def test_measure_mask_length_reports_curve_at_least_axis():
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(mask, (160, 120), (110, 18), 0, 0, 360, 255, -1)

    result = measure_mask_length(mask)

    assert result["comprimento_eixo_px"] > 0.0
    assert result["comprimento_curva_px"] >= result["comprimento_eixo_px"]
