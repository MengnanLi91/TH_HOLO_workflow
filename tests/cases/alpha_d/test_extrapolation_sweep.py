import pytest

from cases.alpha_d.plot_extrapolation_sweep import find_crossover, relerr


def test_relerr():
    assert relerr(11.0, 10.0) == pytest.approx(0.1)
    assert relerr(9.0, 10.0) == pytest.approx(-0.1)


def test_find_crossover_returns_first_axis_value_where_coupled_beats_regressor():
    axis = [0.5, 0.4, 0.3, 0.2]  # decreasing Dr = further out
    regr = [0.02, 0.04, 0.25, 0.60]  # regressor still wins at 0.5 and 0.4
    cpld = [0.05, 0.06, 0.07, 0.09]  # |coupled err| grows slow
    # first axis point where |coupled| < |regr|: at 0.4 regr (0.04) still wins,
    # at 0.3 coupled (0.07) beats regr (0.25) -> crossover 0.3
    assert find_crossover(axis, regr, cpld) == pytest.approx(0.3)


def test_find_crossover_none_when_regressor_always_better():
    axis = [0.5, 0.4, 0.3]
    regr = [0.01, 0.02, 0.03]
    cpld = [0.05, 0.06, 0.07]
    assert find_crossover(axis, regr, cpld) is None
