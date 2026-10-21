import numpy as np
import pytest

from transformations import apply_geometric_adstock, apply_hill_saturation
from optimizer import SpilloverBudgetOptimizer, check_feasibility
from data_generator import get_true_params
from evaluator import SpilloverMMMEvaluator, flatten_params
from data_generator import generate_spillover_mmm_data

CHANNELS = ["tv", "influencer", "meta_ads", "google_search"]
HIST = {"tv": 15.0, "influencer": 6.0, "meta_ads": 15.0, "google_search": 16.0}


def _loop_adstock(spend, alpha):
    out = np.zeros(len(spend))
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + alpha * out[t - 1]
    return out


def test_adstock_matches_loop():
    x = np.random.default_rng(1).uniform(0, 50, 200)
    for a in (0.0, 0.3, 0.9):
        np.testing.assert_allclose(apply_geometric_adstock(x, a), _loop_adstock(x, a))


def test_hill_bounds():
    y = apply_hill_saturation(np.array([0.0, 10.0, 1e6]), 10.0, 1.5)
    assert y[0] == 0 and abs(y[1] - 0.5) < 1e-6 and y[2] < 1.0


def test_param_layout_matches_flatten():
    df = generate_spillover_mmm_data()
    ev = SpilloverMMMEvaluator(
        df, CHANNELS,
        [("tv", "google_search"), ("influencer", "google_search"), ("tv", "meta_ads")],
        [("meta_ads", "google_search")],
    )
    names, x0, bounds = ev._param_layout()
    assert names == list(flatten_params(get_true_params()).keys())
    assert len(names) == len(x0) == len(bounds)


@pytest.fixture(scope="module")
def optimizer():
    return SpilloverBudgetOptimizer(get_true_params(), HIST)


def test_optimizer_respects_budget_and_bounds(optimizer):
    bounds = {"tv": (10, 50), "influencer": (5, 25), "meta_ads": (5, 30), "google_search": (5, 40)}
    res = optimizer.optimize_budget(100.0, bounds)
    alloc = res["optimized_allocation"]
    assert res["success"]
    assert abs(sum(alloc.values()) - 100.0) < 0.05
    for ch, (lo, hi) in bounds.items():
        assert lo - 1e-6 <= alloc[ch] <= hi + 1e-6


def test_optimum_beats_equal_split(optimizer):
    bounds = {ch: (0.0, 100.0) for ch in CHANNELS}
    res = optimizer.optimize_budget(100.0, bounds)
    equal = optimizer.evaluate({ch: 25.0 for ch in CHANNELS})
    assert res["expected_steady_state_sales"] >= equal - 1e-6


def test_mroas_equalized_for_interior_channels(optimizer):
    bounds = {ch: (0.0, 100.0) for ch in CHANNELS}
    res = optimizer.optimize_budget(100.0, bounds)
    alloc = res["optimized_allocation"]
    mr = optimizer.marginal_returns(alloc)
    interior = [mr[ch] for ch in CHANNELS if 0.5 < alloc[ch] < 99.5]
    assert len(interior) >= 2
    assert max(interior) - min(interior) < 0.02


def test_infeasible_constraints_detected(optimizer):
    bounds = {ch: (30.0, 40.0) for ch in CHANNELS}
    assert check_feasibility(100.0, bounds, CHANNELS) is not None
    with pytest.raises(ValueError):
        optimizer.optimize_budget(100.0, bounds)
