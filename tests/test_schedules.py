import math
from itertools import pairwise

import pytest

from model_training.schedules import get_kl_weight

METHODS = ["none", "linear", "exp_annealing", "exp-convex_annealing"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "minimum, maximum",
    [(0.0, 1.0), (0.2, 0.8), (0.0, 0.0), (0.5, 0.5), (0.0, 1e-16), (0.0, 1e308)],
)
def test_weights_are_finite_bounded_monotonic_and_reach_endpoints(
    method, minimum, maximum
):
    values = [
        get_kl_weight(
            step,
            100,
            min_beta=minimum,
            max_beta=maximum,
            method=method,
            warmup_fraction=1.0,
        )
        for step in range(101)
    ]

    assert all(math.isfinite(value) and minimum <= value <= maximum for value in values)
    assert all(left <= right for left, right in pairwise(values))
    assert values[0] == (maximum if method == "none" else minimum)
    assert values[-1] == maximum


@pytest.mark.parametrize("total", [None, -1, 0, math.inf])
@pytest.mark.parametrize("method", METHODS)
def test_unknown_horizon_uses_maximum(total, method):
    assert get_kl_weight(10, total, max_beta=0.8, method=method) == 0.8


def test_linear_warmup_and_clamped_progress():
    options = {
        "min_beta": 0.2,
        "max_beta": 0.8,
        "method": "linear",
        "warmup_fraction": 0.2,
    }
    assert get_kl_weight(-1, 100, **options) == 0.2
    assert get_kl_weight(10, 100, **options) == pytest.approx(0.5)
    assert get_kl_weight(20, 100, **options) == 0.8
    assert get_kl_weight(200, 100, **options) == 0.8
    assert get_kl_weight(0, 100, method="linear", warmup_fraction=0) == 1.0


def test_convex_curve_preserves_geometric_interpolation_for_positive_minimum():
    assert get_kl_weight(
        50,
        100,
        min_beta=0.01,
        max_beta=1.0,
        method="exp-convex_annealing",
        scaling_factor=1,
    ) == pytest.approx(0.1)


@pytest.mark.parametrize("method", ["exp_annealing", "exp-convex_annealing"])
@pytest.mark.parametrize("factor", [1e-12, 1000])
def test_extreme_scaling_factors_remain_bounded(method, factor):
    value = get_kl_weight(50, 100, method=method, scaling_factor=factor)
    assert math.isfinite(value) and 0 <= value <= 1


@pytest.mark.parametrize(
    "options",
    [
        {"method": "typo", "total": -1},
        {"min_beta": -0.1},
        {"min_beta": 2, "max_beta": 1},
        {"max_beta": math.inf},
        {"min_beta": math.nan},
        {"scaling_factor": 0},
        {"scaling_factor": -1},
        {"scaling_factor": math.inf},
        {"method": "linear", "warmup_fraction": -0.1},
        {"method": "linear", "warmup_fraction": 1.1},
        {"method": "linear", "warmup_fraction": math.nan},
        {"current": math.nan},
        {"total": math.nan},
    ],
)
def test_invalid_inputs_raise_instead_of_producing_silent_bad_weights(options):
    with pytest.raises(ValueError):
        get_kl_weight(**{"current": 1, "total": 100, **options})
